import math
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CLASS_NAMES, DOWNSAMPLE, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_OBJECTS, MEAN_DIMS


def parse_calib(path):
    with open(path) as f:
        for line in f:
            if line.startswith("P2:"):
                values = [float(v) for v in line.strip().split(" ")[1:]]
                return np.array(values, dtype=np.float32).reshape(3, 4)
    raise ValueError(f"No P2 line found in {path}")


def parse_labels(path):
    objects = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ")
            if not parts or parts[0] == "DontCare":
                continue
            objects.append(
                {
                    "type": parts[0],
                    "truncated": float(parts[1]),
                    "occluded": int(parts[2]),
                    "alpha": float(parts[3]),
                    "bbox": np.array(parts[4:8], dtype=np.float32),
                    "dims": np.array([parts[8], parts[9], parts[10]], dtype=np.float32),
                    "location": np.array([parts[11], parts[12], parts[13]], dtype=np.float32),
                    "rotation_y": float(parts[14]),
                }
            )
    return objects


def scale_calib(P2, sx, sy):
    P2 = P2.copy()
    P2[0, :] *= sx
    P2[1, :] *= sy
    return P2


def project_point(P2, xyz):
    point = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float32)
    proj = P2 @ point
    return proj[0] / proj[2], proj[1] / proj[2]


def gaussian_radius(width, height, min_overlap=0.7):
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1**2 - 4 * a1 * c1, 0))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2**2 - 4 * a2 * c2, 0))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(b3**2 - 4 * a3 * c3, 0))
    r3 = (b3 + sq3) / 2

    return max(min(r1, r2, r3), 0)


def draw_gaussian(heatmap, center, radius):
    radius = max(int(radius), 0)
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    y_grid, x_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2 + 1e-9))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def encode_depth(z):
    return 1.0 / (1.0 + z)


def build_targets(objects, P2, feat_h, feat_w, stride=DOWNSAMPLE, max_objects=MAX_OBJECTS, mean_dims=MEAN_DIMS):
    heatmap = np.zeros((feat_h, feat_w, len(CLASS_NAMES)), dtype=np.float32)
    indices = np.zeros((max_objects,), dtype=np.int32)
    mask = np.zeros((max_objects,), dtype=np.float32)
    offset = np.zeros((max_objects, 2), dtype=np.float32)
    depth = np.zeros((max_objects, 1), dtype=np.float32)
    dims = np.zeros((max_objects, 3), dtype=np.float32)
    orientation = np.zeros((max_objects, 2), dtype=np.float32)

    mean_dims = np.array(mean_dims, dtype=np.float32)
    num_objects = 0

    for obj in objects:
        if obj["type"] not in CLASS_NAMES or num_objects >= max_objects:
            continue

        h, w, l = obj["dims"]
        x, y, z = obj["location"]
        if z <= 0:
            continue
        centroid = np.array([x, y - h / 2.0, z], dtype=np.float32)
        u, v = project_point(P2, centroid)
        fx, fy = u / stride, v / stride
        if not (0 <= fx < feat_w and 0 <= fy < feat_h):
            continue

        box_w = (obj["bbox"][2] - obj["bbox"][0]) / stride
        box_h = (obj["bbox"][3] - obj["bbox"][1]) / stride
        radius = gaussian_radius(box_w, box_h)
        draw_gaussian(heatmap[:, :, CLASS_NAMES.index(obj["type"])], (fx, fy), radius)

        xi, yi = int(fx), int(fy)
        indices[num_objects] = yi * feat_w + xi
        mask[num_objects] = 1.0
        offset[num_objects] = [fx - xi, fy - yi]
        depth[num_objects] = encode_depth(z)
        dims[num_objects] = np.log(np.array([h, w, l], dtype=np.float32) / mean_dims)
        orientation[num_objects] = [math.sin(obj["alpha"]), math.cos(obj["alpha"])]
        num_objects += 1

    return {
        "heatmap": heatmap,
        "indices": indices,
        "mask": mask,
        "offset": offset,
        "depth": depth,
        "dims": dims,
        "orientation": orientation,
    }


def load_image_and_objects(image_path, label_path, calib_path, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    target_h, target_w = image_size
    image = image.resize((target_w, target_h), Image.BILINEAR)
    image_arr = np.asarray(image, dtype=np.float32) / 255.0

    sx, sy = target_w / orig_w, target_h / orig_h
    P2 = scale_calib(parse_calib(calib_path), sx, sy)
    objects = parse_labels(label_path)
    return image_arr, P2, objects


def load_sample(image_path, label_path, calib_path, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), stride=DOWNSAMPLE):
    image_arr, P2, objects = load_image_and_objects(image_path, label_path, calib_path, image_size)
    target_h, target_w = image_size
    targets = build_targets(objects, P2, target_h // stride, target_w // stride, stride=stride)
    return image_arr, targets


def list_sample_ids(root_dir, split="training"):
    label_dir = os.path.join(root_dir, split, "label_2")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".txt"))


def train_val_split(sample_ids, val_fraction=0.1, seed=0, mapping_path=None):
    rng = random.Random(seed)

    if mapping_path and os.path.exists(mapping_path) and os.path.getsize(mapping_path) > 0:
        with open(mapping_path) as f:
            lines = [line.strip().split(" ") for line in f if line.strip()]
        sequence_by_id = {f"{i:06d}": f"{date}_{drive}" for i, (date, drive, _frame) in enumerate(lines)}

        sequences = sorted({sequence_by_id[sid] for sid in sample_ids if sid in sequence_by_id})
        rng.shuffle(sequences)
        num_val_sequences = max(1, round(len(sequences) * val_fraction))
        val_sequences = set(sequences[:num_val_sequences])

        train_ids = [sid for sid in sample_ids if sequence_by_id.get(sid) not in val_sequences]
        val_ids = [sid for sid in sample_ids if sequence_by_id.get(sid) in val_sequences]
        return train_ids, val_ids

    shuffled = list(sample_ids)
    rng.shuffle(shuffled)
    num_val = max(1, round(len(shuffled) * val_fraction))
    return shuffled[num_val:], shuffled[:num_val]


def load_kitti_dataset(root_dir, split="training", batch_size=4, shuffle=True, sample_ids=None):
    sample_ids = sample_ids if sample_ids is not None else list_sample_ids(root_dir, split)

    def generator():
        for sample_id in sample_ids:
            image_path = os.path.join(root_dir, split, "image_2", f"{sample_id}.png")
            label_path = os.path.join(root_dir, split, "label_2", f"{sample_id}.txt")
            calib_path = os.path.join(root_dir, split, "calib", f"{sample_id}.txt")
            yield load_sample(image_path, label_path, calib_path)

    feat_h, feat_w = IMAGE_HEIGHT // DOWNSAMPLE, IMAGE_WIDTH // DOWNSAMPLE
    output_signature = (
        tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32),
        {
            "heatmap": tf.TensorSpec(shape=(feat_h, feat_w, len(CLASS_NAMES)), dtype=tf.float32),
            "indices": tf.TensorSpec(shape=(MAX_OBJECTS,), dtype=tf.int32),
            "mask": tf.TensorSpec(shape=(MAX_OBJECTS,), dtype=tf.float32),
            "offset": tf.TensorSpec(shape=(MAX_OBJECTS, 2), dtype=tf.float32),
            "depth": tf.TensorSpec(shape=(MAX_OBJECTS, 1), dtype=tf.float32),
            "dims": tf.TensorSpec(shape=(MAX_OBJECTS, 3), dtype=tf.float32),
            "orientation": tf.TensorSpec(shape=(MAX_OBJECTS, 2), dtype=tf.float32),
        },
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(sample_ids), 1000))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
