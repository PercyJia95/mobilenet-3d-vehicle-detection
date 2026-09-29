
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from PIL import Image

from detector.config import IMAGE_HEIGHT, IMAGE_WIDTH
from detector.decode import decode_predictions
from detector.kitti import load_sample, parse_calib, scale_calib
from detector.losses import compute_losses
from detector.model import build_3d_detector

CALIB_LINE = (
    "P2: 7.070912e+02 0.000000e+00 6.018873e+02 4.688783e+01 "
    "0.000000e+00 7.070912e+02 1.831104e+02 1.178601e-01 "
    "0.000000e+00 0.000000e+00 1.000000e+00 6.203223e-03\n"
)
LABEL_LINE = "Car 0.00 0 -1.57 599.41 156.40 629.75 189.25 1.49 1.63 3.88 0.47 1.49 12.34 -1.56\n"


def make_synthetic_sample(root):
    for sub in ("image_2", "label_2", "calib"):
        os.makedirs(os.path.join(root, "training", sub), exist_ok=True)

    image = Image.fromarray((np.random.rand(370, 1224, 3) * 255).astype("uint8"))
    image.save(os.path.join(root, "training", "image_2", "000000.png"))

    with open(os.path.join(root, "training", "label_2", "000000.txt"), "w") as f:
        f.write(LABEL_LINE)

    with open(os.path.join(root, "training", "calib", "000000.txt"), "w") as f:
        f.write(CALIB_LINE)


def main():
    tmp_root = tempfile.mkdtemp(prefix="kitti_sanity_")
    try:
        make_synthetic_sample(tmp_root)

        image, targets = load_sample(
            os.path.join(tmp_root, "training", "image_2", "000000.png"),
            os.path.join(tmp_root, "training", "label_2", "000000.txt"),
            os.path.join(tmp_root, "training", "calib", "000000.txt"),
        )
        print("Parsed sample -> image:", image.shape, "heatmap:", targets["heatmap"].shape)
        assert targets["mask"].sum() == 1.0, "expected exactly one labeled object"
        print("Target encoding OK, 1 object placed at heatmap index", int(targets["indices"][0]))

        batch_images = tf.constant(image[None, ...])
        batch_targets = {k: tf.constant(v[None, ...]) for k, v in targets.items()}

        model = build_3d_detector(pretrained_backbone=True)
        predictions = model(batch_images, training=False)
        total_loss, loss_terms = compute_losses(predictions, batch_targets)
        print("Forward pass OK, loss terms:", {k: float(v) for k, v in loss_terms.items()})

        optimizer = tf.keras.optimizers.Adam(1e-3)
        with tf.GradientTape() as tape:
            predictions = model(batch_images, training=True)
            total_loss, _ = compute_losses(predictions, batch_targets)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Gradient step OK, total_loss={float(total_loss):.4f}")

        orig_w, orig_h = 1224, 370
        P2 = scale_calib(
            parse_calib(os.path.join(tmp_root, "training", "calib", "000000.txt")),
            IMAGE_WIDTH / orig_w,
            IMAGE_HEIGHT / orig_h,
        )
        single_pred = {k: v[0] for k, v in predictions.items()}
        detections = decode_predictions(single_pred, P2, score_thresh=0.0, max_dets=1)
        print(f"Decode OK, produced {len(detections)} detection(s) (untrained model, scores are meaningless)")

        print("Sanity check passed.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
