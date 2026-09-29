import numpy as np
import tensorflow as tf

from .config import DOWNSAMPLE, MEAN_DIMS


def _nms_peaks(heatmap, kernel=3):
    pooled = tf.nn.max_pool2d(heatmap[tf.newaxis], ksize=kernel, strides=1, padding="SAME")[0]
    keep = tf.cast(tf.equal(heatmap, pooled), heatmap.dtype)
    return heatmap * keep


def decode_depth(encoded):
    return 1.0 / np.clip(encoded, 1e-4, 1.0) - 1.0


def decode_predictions(predictions, P2, stride=DOWNSAMPLE, score_thresh=0.3, max_dets=20, mean_dims=MEAN_DIMS):
    heatmap = tf.sigmoid(predictions["heatmap"])
    heatmap = _nms_peaks(heatmap)
    height, width, num_classes = heatmap.shape

    scores = heatmap.numpy().reshape(-1)
    offset_np = predictions["offset"].numpy()
    depth_np = tf.sigmoid(predictions["depth"]).numpy()
    dims_np = predictions["dims"].numpy()
    orientation_np = predictions["orientation"].numpy()

    order = np.argsort(-scores)[:max_dets]

    fx, fy = P2[0, 0], P2[1, 1]
    cx, cy = P2[0, 2], P2[1, 2]
    mean_dims = np.array(mean_dims, dtype=np.float32)

    detections = []
    for flat_idx in order:
        score = scores[flat_idx]
        if score < score_thresh:
            break

        yx = flat_idx // num_classes
        class_id = flat_idx % num_classes
        yi, xi = divmod(yx, width)

        off = offset_np[yi, xi]
        u = (xi + off[0]) * stride
        v = (yi + off[1]) * stride

        z = decode_depth(depth_np[yi, xi, 0])
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy

        dims_pred = dims_np[yi, xi]
        h, w, l = mean_dims * np.exp(dims_pred)

        sin_a, cos_a = orientation_np[yi, xi]
        alpha = np.arctan2(sin_a, cos_a)
        rotation_y = alpha + np.arctan2(x_cam, z)

        location = np.array([x_cam, y_cam + h / 2.0, z], dtype=np.float32)

        detections.append(
            {
                "score": float(score),
                "class_id": int(class_id),
                "location": location,
                "dims": np.array([h, w, l], dtype=np.float32),
                "rotation_y": float(rotation_y),
            }
        )

    return detections
