import math
import os

import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH
from .decode import decode_predictions
from .kitti import load_image_and_objects


def _box_corners_bev(x, z, w, l, rotation_y):
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2])
    dz = np.array([w / 2, -w / 2, -w / 2, w / 2])
    cos_r, sin_r = math.cos(rotation_y), math.sin(rotation_y)
    corners_x = x + dx * cos_r + dz * sin_r
    corners_z = z - dx * sin_r + dz * cos_r
    return list(zip(corners_x, corners_z))


def _polygon_area(polygon):
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _clip_polygon(subject, clip_edge_start, clip_edge_end):

    def inside(p):
        edge = (clip_edge_end[0] - clip_edge_start[0], clip_edge_end[1] - clip_edge_start[1])
        rel = (p[0] - clip_edge_start[0], p[1] - clip_edge_start[1])
        return edge[0] * rel[1] - edge[1] * rel[0] <= 0

    def intersect(p1, p2):
        edge = (clip_edge_end[0] - clip_edge_start[0], clip_edge_end[1] - clip_edge_start[1])
        diff = (p1[0] - p2[0], p1[1] - p2[1])
        denom = edge[0] * diff[1] - edge[1] * diff[0]
        if abs(denom) < 1e-12:
            return p2
        t = (edge[0] * (p1[1] - clip_edge_start[1]) - edge[1] * (p1[0] - clip_edge_start[0])) / denom
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    output = []
    n = len(subject)
    for i in range(n):
        current, previous = subject[i], subject[i - 1]
        if inside(current):
            if not inside(previous):
                output.append(intersect(previous, current))
            output.append(current)
        elif inside(previous):
            output.append(intersect(previous, current))
    return output


def _convex_intersection_area(poly_a, poly_b):
    output = poly_a
    n = len(poly_b)
    for i in range(n):
        if not output:
            return 0.0
        output = _clip_polygon(output, poly_b[i - 1], poly_b[i])
    return _polygon_area(output) if output else 0.0


def bev_iou(box_a, box_b):
    xa, _, za = box_a["location"]
    ha, wa, la = box_a["dims"]
    xb, _, zb = box_b["location"]
    hb, wb, lb = box_b["dims"]

    corners_a = _box_corners_bev(xa, za, wa, la, box_a["rotation_y"])
    corners_b = _box_corners_bev(xb, zb, wb, lb, box_b["rotation_y"])

    intersection = _convex_intersection_area(corners_a, corners_b)
    area_a, area_b = wa * la, wb * lb
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def iou_3d(box_a, box_b):
    xa, ya, za = box_a["location"]
    ha, wa, la = box_a["dims"]
    xb, yb, zb = box_b["location"]
    hb, wb, lb = box_b["dims"]

    corners_a = _box_corners_bev(xa, za, wa, la, box_a["rotation_y"])
    corners_b = _box_corners_bev(xb, zb, wb, lb, box_b["rotation_y"])
    bev_intersection = _convex_intersection_area(corners_a, corners_b)

    top_a, top_b = ya - ha, yb - hb
    height_overlap = max(0.0, min(ya, yb) - max(top_a, top_b))

    intersection = bev_intersection * height_overlap
    volume_a, volume_b = ha * wa * la, hb * wb * lb
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(scores, matches, num_ground_truth, num_points=40):
    if num_ground_truth == 0:
        return 0.0
    if len(scores) == 0:
        return 0.0

    order = np.argsort(-np.asarray(scores))
    matches = np.asarray(matches)[order]

    true_positives = np.cumsum(matches)
    false_positives = np.cumsum(1 - matches)
    recall = true_positives / num_ground_truth
    precision = true_positives / (true_positives + false_positives + 1e-9)

    recall_levels = np.linspace(0, 1, num_points + 1)[1:]
    interpolated = np.zeros_like(recall_levels)
    for i, r in enumerate(recall_levels):
        mask = recall >= r
        interpolated[i] = precision[mask].max() if mask.any() else 0.0

    return float(interpolated.mean())


def match_detections(all_detections, all_ground_truths, iou_fn, iou_threshold):
    scores, tp_flags = [], []
    num_ground_truth = sum(len(gts) for gts in all_ground_truths)

    for detections, ground_truths in zip(all_detections, all_ground_truths):
        matched = [False] * len(ground_truths)
        for det in sorted(detections, key=lambda d: -d["score"]):
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(ground_truths):
                if matched[i]:
                    continue
                iou = iou_fn(det, gt)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            scores.append(det["score"])
            if best_iou >= iou_threshold:
                matched[best_idx] = True
                tp_flags.append(1)
            else:
                tp_flags.append(0)

    return scores, tp_flags, num_ground_truth


def orientation_error_degrees(det, gt):
    diff = abs(det["rotation_y"] - gt["rotation_y"]) % (2 * math.pi)
    diff = min(diff, 2 * math.pi - diff)
    return math.degrees(diff)


def evaluate(
    model,
    data_root,
    sample_ids,
    split="training",
    iou_threshold=0.7,
    score_thresh=0.1,
    max_dets=50,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
):
    all_detections, all_ground_truths = [], []
    depth_errors, orientation_errors = [], []

    for sample_id in sample_ids:
        image_path = os.path.join(data_root, split, "image_2", f"{sample_id}.png")
        label_path = os.path.join(data_root, split, "label_2", f"{sample_id}.txt")
        calib_path = os.path.join(data_root, split, "calib", f"{sample_id}.txt")

        image, P2, objects = load_image_and_objects(image_path, label_path, calib_path, image_size)
        ground_truths = [obj for obj in objects if obj["type"] in CLASS_NAMES]

        predictions = model(image[tf.newaxis, ...], training=False)
        predictions = {k: v[0] for k, v in predictions.items()}
        detections = decode_predictions(predictions, P2, score_thresh=score_thresh, max_dets=max_dets)

        all_detections.append(detections)
        all_ground_truths.append(ground_truths)

    metrics = {}
    for name, iou_fn in (("BEV", bev_iou), ("3D", iou_3d)):
        scores, tp_flags, num_gt = match_detections(all_detections, all_ground_truths, iou_fn, iou_threshold)
        metrics[f"AP_{name}"] = average_precision(scores, tp_flags, num_gt)

    for detections, ground_truths in zip(all_detections, all_ground_truths):
        matched = [False] * len(ground_truths)
        for det in sorted(detections, key=lambda d: -d["score"]):
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(ground_truths):
                if matched[i]:
                    continue
                iou = iou_3d(det, gt)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= iou_threshold:
                matched[best_idx] = True
                depth_errors.append(abs(det["location"][2] - ground_truths[best_idx]["location"][2]))
                orientation_errors.append(orientation_error_degrees(det, ground_truths[best_idx]))

    metrics["depth_MAE_m"] = float(np.mean(depth_errors)) if depth_errors else float("nan")
    metrics["orientation_error_deg"] = float(np.mean(orientation_errors)) if orientation_errors else float("nan")
    metrics["num_ground_truth"] = sum(len(gts) for gts in all_ground_truths)
    metrics["num_detections"] = sum(len(dets) for dets in all_detections)

    return metrics
