"""Export a trained detector to a single-output TFLite model for on-device (iOS) inference.

The exported model takes one RGB image and returns one table of the top-K candidates:

    input  "image":      float32 [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3], RGB in [0, 1]
    output "detections": float32 [1, TOP_K, 12], rows sorted by score (descending), columns = TABLE_COLUMNS

Sigmoid, 3x3 max-pool peak picking, top-K and gathering run inside the model. The app only thresholds
the score and does the camera-dependent 3D decoding, mirrored here by decode_table().
"""

import argparse
import json
import math
import os
import shutil
import tempfile
import warnings

import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, DOWNSAMPLE, IMAGE_HEIGHT, IMAGE_WIDTH, MEAN_DIMS
from .decode import decode_predictions
from .kitti import list_sample_ids, load_image_and_objects, train_val_split

TOP_K = 50
TABLE_COLUMNS = [
    "score",  # sigmoid heatmap peak value
    "x",  # feature-map column of the peak (integer-valued)
    "y",  # feature-map row of the peak (integer-valued)
    "offset_x",  # sub-cell offset added to x
    "offset_y",  # sub-cell offset added to y
    "depth",  # sigmoid depth encoding: z = 1 / depth - 1
    "dim_h",  # log(h / mean_h)
    "dim_w",  # log(w / mean_w)
    "dim_l",  # log(l / mean_l)
    "sin_alpha",  # observation angle, alpha = atan2(sin, cos)
    "cos_alpha",
    "class_id",
]
COL = {name: i for i, name in enumerate(TABLE_COLUMNS)}


def build_detect_fn(model, top_k=TOP_K):
    feat_h, feat_w = IMAGE_HEIGHT // DOWNSAMPLE, IMAGE_WIDTH // DOWNSAMPLE
    num_classes = len(CLASS_NAMES)

    def detect(image):
        outputs = model(image, training=False)

        # Peak picking, as in decode.py: keep cells that are the maximum of their 3x3 neighbourhood.
        heatmap = tf.sigmoid(outputs["heatmap"])
        pooled = tf.nn.max_pool2d(heatmap, ksize=3, strides=1, padding="SAME")
        peaks = heatmap * tf.cast(tf.equal(heatmap, pooled), heatmap.dtype)

        scores, flat_idx = tf.math.top_k(tf.reshape(peaks, [-1]), k=top_k)
        class_id = flat_idx % num_classes
        cell = flat_idx // num_classes
        y, x = cell // feat_w, cell % feat_w

        def at_peaks(feature_map):
            flat = tf.reshape(feature_map, [feat_h * feat_w, feature_map.shape[-1]])
            return tf.gather(flat, cell)

        table = tf.concat(
            [
                scores[:, None],
                tf.cast(x, tf.float32)[:, None],
                tf.cast(y, tf.float32)[:, None],
                at_peaks(outputs["offset"]),
                at_peaks(tf.sigmoid(outputs["depth"])),
                at_peaks(outputs["dims"]),
                at_peaks(outputs["orientation"]),
                tf.cast(class_id, tf.float32)[:, None],
            ],
            axis=-1,
        )
        return {"detections": table[None]}

    return detect


def export_saved_model(model, path, top_k=TOP_K):
    archive = tf.keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint(
        name="serving_default",
        fn=build_detect_fn(model, top_k),
        input_signature=[tf.TensorSpec([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3], tf.float32, name="image")],
    )
    archive.write_out(path, verbose=False)


def convert_to_tflite(saved_model_path, float16=True):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path, signature_keys=["serving_default"])
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def decode_table(table, P2, score_thresh=0.3, stride=DOWNSAMPLE, mean_dims=MEAN_DIMS):
    """Reference for the on-device decoder: plain arithmetic only, so it ports line by line to Swift.

    P2 is the 3x4 projection matrix of the model-input image (IMAGE_WIDTH x IMAGE_HEIGHT).
    """
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    detections = []
    for row in table:
        score = row[COL["score"]]
        if score < score_thresh:
            break  # rows are sorted by score

        u = (row[COL["x"]] + row[COL["offset_x"]]) * stride
        v = (row[COL["y"]] + row[COL["offset_y"]]) * stride
        z = 1.0 / min(max(row[COL["depth"]], 1e-4), 1.0) - 1.0
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy

        h = mean_dims[0] * math.exp(row[COL["dim_h"]])
        w = mean_dims[1] * math.exp(row[COL["dim_w"]])
        l = mean_dims[2] * math.exp(row[COL["dim_l"]])

        alpha = math.atan2(row[COL["sin_alpha"]], row[COL["cos_alpha"]])
        rotation_y = alpha + math.atan2(x_cam, z)

        detections.append(
            {
                "score": float(score),
                "class_id": int(round(row[COL["class_id"]])),
                "location": [float(x_cam), float(y_cam + h / 2.0), float(z)],
                "dims": [float(h), float(w), float(l)],
                "rotation_y": float(rotation_y),
            }
        )
    return detections


class TFLiteDetector:
    def __init__(self, model_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tf.lite.Interpreter deprecation notice
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        (self.input_detail,) = self.interpreter.get_input_details()
        (self.output_detail,) = self.interpreter.get_output_details()

    def __call__(self, image):
        self.interpreter.set_tensor(self.input_detail["index"], image[None].astype(np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_detail["index"])[0]


def _angle_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _box_errors(ref, cand):
    location = np.linalg.norm(np.array(cand["location"]) - np.array(ref["location"])) / max(ref["location"][2], 1.0)
    return {
        "score": abs(cand["score"] - ref["score"]),
        "location_rel": float(location),
        "dims": float(np.abs(np.array(cand["dims"]) - np.array(ref["dims"])).max()),
        "rotation_y": _angle_diff(cand["rotation_y"], ref["rotation_y"]),
    }


def _merge_worst(worst, errors):
    for k, v in errors.items():
        worst[k] = max(worst.get(k, 0.0), v)


# Layer 1 compares identical float32 computations, so anything above rounding noise is a logic bug
# (column order, x/y, formulas). Layer 2 compares the converted model against the same graph in TF.
LOGIC_TOLERANCES = {"score": 1e-5, "location_rel": 1e-4, "dims": 1e-4, "rotation_y": 1e-4}
CONVERSION_TOLERANCES = {
    "float32": {"score": 1e-4, "location_rel": 1e-3, "dims": 1e-3, "rotation_y": 1e-3},
    # float16 weights shift outputs slightly; observed on run1: score 0.015, location 1.1%, rotation 2.2 deg.
    "float16": {"score": 0.03, "location_rel": 0.03, "dims": 0.05, "rotation_y": 0.1},
}
# A peak this close to a 3x3 neighbour (or to the score threshold) can flip under any numeric change,
# including different CPU/GPU kernels on the phone, so it may legitimately appear on one side only.
FRAGILE_MARGIN = 0.03


def check_logic(reference, candidate):
    """Keras + decode.py vs the unconverted TF graph + decode_table: should agree to rounding noise."""
    worst, unmatched = {}, 0
    remaining = list(candidate)
    for ref in reference:
        dists = [np.linalg.norm(np.array(c["location"]) - np.array(ref["location"])) for c in remaining]
        if not dists or min(dists) > 1e-3 * max(ref["location"][2], 1.0):
            unmatched += 1
            continue
        _merge_worst(worst, _box_errors(ref, remaining.pop(int(np.argmin(dists)))))
    return worst, unmatched + len(remaining)


def _peak_margin(heatmap, x, y, class_id):
    """How far a cell is above (positive) or below (negative) the best of its 3x3 neighbours."""
    h, w = heatmap.shape[:2]
    patch = heatmap[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2, class_id].copy()
    value = heatmap[y, x, class_id]
    patch[min(y, 1), min(x, 1)] = -np.inf
    return float(value - patch.max())


def check_conversion(tf_table, tflite_table, heatmap, P2, score_thresh):
    """Match rows of the two tables by peak cell, then compare the decoded boxes."""
    def cells(table):
        rows = {}
        for row in table:
            if row[COL["score"]] >= score_thresh - FRAGILE_MARGIN:
                rows[(int(row[COL["x"]]), int(row[COL["y"]]), int(round(row[COL["class_id"]])))] = row
        return rows

    tf_rows, lite_rows = cells(tf_table), cells(tflite_table)
    worst, real_misses, fragile_misses = {}, 0, 0
    for key in set(tf_rows) | set(lite_rows):
        in_tf, in_lite = key in tf_rows, key in lite_rows
        if in_tf and in_lite:
            ref = decode_table(tf_rows[key][None], P2, score_thresh=-1.0)[0]
            cand = decode_table(lite_rows[key][None], P2, score_thresh=-1.0)[0]
            _merge_worst(worst, _box_errors(ref, cand))
            continue
        row = tf_rows[key] if in_tf else lite_rows[key]
        if row[COL["score"]] < score_thresh + FRAGILE_MARGIN:
            continue  # near the threshold: may sit on either side
        if abs(_peak_margin(heatmap, *key)) < FRAGILE_MARGIN:
            fragile_misses += 1
        else:
            real_misses += 1
    return worst, real_misses, fragile_misses


def _report(title, worst, tolerances, missing, extra):
    ok = missing == 0 and all(worst.get(k, 0.0) <= tol for k, tol in tolerances.items())
    print(
        f"  {title}: {'PASS' if ok else 'FAIL'} - worst score {worst.get('score', 0):.2g}, "
        f"location {worst.get('location_rel', 0):.2g} x depth, dims {worst.get('dims', 0):.2g} m, "
        f"rotation {math.degrees(worst.get('rotation_y', 0)):.2g} deg{extra}"
    )
    return ok


def verify(model, tflite_path, data_root, sample_ids, score_thresh, float16):
    detect = tf.function(build_detect_fn(model))
    detector = TFLiteDetector(tflite_path)
    logic_worst, conv_worst = {}, {}
    logic_unmatched = real_misses = fragile_misses = total_boxes = 0
    per_sample = []
    for sample_id in sample_ids:
        image, P2, _ = load_image_and_objects(*_sample_paths(data_root, sample_id))
        keras_out = model(image[None], training=False)
        reference = decode_predictions(
            {k: v[0] for k, v in keras_out.items()}, P2, score_thresh=score_thresh, max_dets=TOP_K
        )
        total_boxes += len(reference)

        tf_table = detect(tf.constant(image[None]))["detections"][0].numpy()
        worst, unmatched = check_logic(reference, decode_table(tf_table, P2, score_thresh))
        _merge_worst(logic_worst, worst)
        logic_unmatched += unmatched

        tflite_table = detector(image)
        heatmap = tf.sigmoid(keras_out["heatmap"])[0].numpy()
        worst, real, fragile = check_conversion(tf_table, tflite_table, heatmap, P2, score_thresh)
        _merge_worst(conv_worst, worst)
        real_misses += real
        fragile_misses += fragile

        per_sample.append((sample_id, image, P2, tflite_table, decode_table(tflite_table, P2, score_thresh)))

    precision = "float16" if float16 else "float32"
    print(f"Verified on {len(sample_ids)} validation images ({total_boxes} boxes from Keras + decode.py)")
    logic_ok = _report(
        "1. post-processing logic (TF graph + decode_table vs decode.py)",
        logic_worst,
        LOGIC_TOLERANCES,
        logic_unmatched,
        f", unmatched boxes {logic_unmatched}",
    )
    conv_ok = _report(
        f"2. {precision} conversion (TFLite vs TF graph)",
        conv_worst,
        CONVERSION_TOLERANCES[precision],
        real_misses,
        f", missing peaks {real_misses} (+{fragile_misses} near-tie peaks, allowed)",
    )
    return logic_ok and conv_ok, per_sample


def _sample_paths(data_root, sample_id):
    return (
        os.path.join(data_root, "training", "image_2", f"{sample_id}.png"),
        os.path.join(data_root, "training", "label_2", f"{sample_id}.txt"),
        os.path.join(data_root, "training", "calib", f"{sample_id}.txt"),
    )


def write_golden(out_dir, data_root, per_sample, score_thresh, count):
    """Save test cases for the app: the exact input tensor, the source image + calibration, and the
    expected TFLite output (raw table and decoded boxes)."""
    golden_dir = os.path.join(out_dir, "golden")
    os.makedirs(golden_dir, exist_ok=True)
    # Prefer busy scenes so the test exercises several boxes.
    chosen = sorted(per_sample, key=lambda s: -len(s[4]))[:count]
    for sample_id, image, P2, table, detections in chosen:
        image_path, _, calib_path = _sample_paths(data_root, sample_id)
        image.astype("<f4").tofile(os.path.join(golden_dir, f"{sample_id}_input.bin"))
        shutil.copy(image_path, os.path.join(golden_dir, f"{sample_id}.png"))
        shutil.copy(calib_path, os.path.join(golden_dir, f"{sample_id}_calib.txt"))
        with open(os.path.join(golden_dir, f"{sample_id}_expected.json"), "w") as f:
            json.dump(
                {
                    "sample_id": sample_id,
                    "input_file": f"{sample_id}_input.bin",
                    "input_format": f"float32 little-endian, NHWC [1, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3], RGB in [0, 1]",
                    "P2_model_input": P2.tolist(),
                    "score_thresh": score_thresh,
                    "table": table.tolist(),
                    "detections": detections,
                },
                f,
                indent=1,
            )
        print(f"  golden case {sample_id}: {len(detections)} detection(s)")


def write_metadata(out_dir, tflite_name, float16, score_thresh):
    meta = {
        "model_file": tflite_name,
        "precision": "float16 weights" if float16 else "float32",
        "input": {"name": "image", "shape": [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3], "dtype": "float32", "format": "RGB in [0, 1]"},
        "output": {"name": "detections", "shape": [1, TOP_K, len(TABLE_COLUMNS)], "dtype": "float32", "sorted_by": "score desc"},
        "columns": TABLE_COLUMNS,
        "class_names": CLASS_NAMES,
        "stride": DOWNSAMPLE,
        "mean_dims_hwl": list(MEAN_DIMS),
        "default_score_thresh": score_thresh,
        "decoding": [
            "u = (x + offset_x) * stride;  v = (y + offset_y) * stride",
            "z = 1 / clamp(depth, 1e-4, 1) - 1",
            "X = (u - cx) * z / fx;  Y = (v - cy) * z / fy   (intrinsics of the model-input image)",
            "h, w, l = mean_dims_hwl * exp(dim_h, dim_w, dim_l)",
            "rotation_y = atan2(sin_alpha, cos_alpha) + atan2(X, z)",
            "box bottom-centre location = (X, Y + h / 2, z)",
        ],
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description="Export the 3D detector to TFLite, verify it, and write golden test files.")
    parser.add_argument("--checkpoint", required=True, help="Trained .keras model")
    parser.add_argument("--data_root", required=True, help="KITTI root dir containing training/{image_2,label_2,calib}")
    parser.add_argument("--out_dir", default="export")
    parser.add_argument("--float32", action="store_true", help="Keep float32 weights (default: float16, half the size)")
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--num_verify", type=int, default=20, help="Validation images used to verify the export")
    parser.add_argument("--num_golden", type=int, default=1, help="Golden test cases to write (~6 MB each)")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Must match training, so samples are held out")
    parser.add_argument("--seed", type=int, default=0, help="Must match training, so samples are held out")
    parser.add_argument("--mapping_path", default=None, help="Must match training, so samples are held out")
    args = parser.parse_args()

    float16 = not args.float32
    os.makedirs(args.out_dir, exist_ok=True)

    # compile=False: load the network only, without the optimizer state stored for resuming training.
    model = tf.keras.models.load_model(args.checkpoint, compile=False)

    with tempfile.TemporaryDirectory() as tmp:
        saved_model_path = os.path.join(tmp, "saved_model")
        export_saved_model(model, saved_model_path)
        tflite_bytes = convert_to_tflite(saved_model_path, float16=float16)

    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    tflite_name = f"{stem}_{'fp16' if float16 else 'fp32'}.tflite"
    tflite_path = os.path.join(args.out_dir, tflite_name)
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"Wrote {tflite_path} ({len(tflite_bytes) / 1e6:.1f} MB)")

    _, val_ids = train_val_split(list_sample_ids(args.data_root, "training"), args.val_fraction, args.seed, args.mapping_path)
    verify_ids = sorted(val_ids)[: args.num_verify]
    passed, per_sample = verify(model, tflite_path, args.data_root, verify_ids, args.score_thresh, float16)

    write_metadata(args.out_dir, tflite_name, float16, args.score_thresh)
    if args.num_golden:
        write_golden(args.out_dir, args.data_root, per_sample, args.score_thresh, args.num_golden)
    print(f"Wrote {os.path.join(args.out_dir, 'model_meta.json')}")

    if not passed:
        raise SystemExit("Export verification FAILED: the TFLite model does not match the Keras model.")


if __name__ == "__main__":
    main()
