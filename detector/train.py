import argparse
import math
import os

import tensorflow as tf

from .kitti import list_sample_ids, load_kitti_dataset, train_val_split
from .losses import compute_losses
from .metrics import evaluate
from .model import build_3d_detector


def _epoch_state_path(checkpoint_path):
    return f"{checkpoint_path}.epoch"


def build_optimizer(learning_rate, warmup_steps, clipnorm):
    # Linear warmup from 0 to learning_rate, then constant (alpha=1.0 disables the cosine decay).
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=1,
        alpha=1.0,
        warmup_target=learning_rate,
        warmup_steps=warmup_steps,
    )
    return tf.keras.optimizers.Adam(learning_rate=schedule, global_clipnorm=clipnorm or None)


def make_train_step(model, optimizer):
    @tf.function
    def train_step(images, targets):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            total_loss, loss_terms = compute_losses(predictions, targets)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss, loss_terms

    return train_step


def make_val_step(model):
    @tf.function
    def val_step(images, targets):
        predictions = model(images, training=False)
        return compute_losses(predictions, targets)

    return val_step


def _log_scalars(writer, loss_terms, total_loss, step, prefix):
    with writer.as_default():
        tf.summary.scalar(f"{prefix}/loss_total", total_loss, step=step)
        for name, value in loss_terms.items():
            tf.summary.scalar(f"{prefix}/loss_{name}", value, step=step)


def _log_heatmap_sample(writer, model, val_ids, data_root, step):
    from .kitti import load_sample

    sample_id = val_ids[0]
    image, targets = load_sample(
        os.path.join(data_root, "training", "image_2", f"{sample_id}.png"),
        os.path.join(data_root, "training", "label_2", f"{sample_id}.txt"),
        os.path.join(data_root, "training", "calib", f"{sample_id}.txt"),
    )
    predictions = model(image[tf.newaxis, ...], training=False)
    pred_heatmap = tf.sigmoid(predictions["heatmap"])
    gt_heatmap = targets["heatmap"][tf.newaxis, ...]

    with writer.as_default():
        tf.summary.image("val/heatmap_pred", pred_heatmap, step=step)
        tf.summary.image("val/heatmap_gt", gt_heatmap, step=step)


def train(
    data_root,
    epochs=10,
    batch_size=4,
    learning_rate=1e-4,
    warmup_steps=500,
    clipnorm=10.0,
    pretrained_backbone=True,
    checkpoint_path=None,
    limit=None,
    val_fraction=0.1,
    mapping_path=None,
    seed=0,
    log_dir="logs",
    map_every=1,
):
    sample_ids = list_sample_ids(data_root, "training")
    if limit:
        sample_ids = sample_ids[:limit]
    train_ids, val_ids = train_val_split(sample_ids, val_fraction=val_fraction, seed=seed, mapping_path=mapping_path)
    print(f"Train/val split: {len(train_ids)} train, {len(val_ids)} val samples")

    train_dataset = load_kitti_dataset(data_root, split="training", batch_size=batch_size, sample_ids=train_ids)
    val_dataset = load_kitti_dataset(
        data_root, split="training", batch_size=batch_size, sample_ids=val_ids, shuffle=False
    )
    steps_per_epoch = math.ceil(len(train_ids) / batch_size)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)
        state_path = _epoch_state_path(checkpoint_path)
        if os.path.exists(state_path):
            with open(state_path) as f:
                start_epoch = int(f.read().strip())
        print(f"Resumed from {checkpoint_path} (completed {start_epoch} epoch(s) so far)")
    else:
        model = build_3d_detector(pretrained_backbone=pretrained_backbone)

    if start_epoch >= epochs:
        print(f"Checkpoint already has {start_epoch} epochs >= target {epochs}; nothing to do.")
        return model

    # Compiling attaches the optimizer to the model, so model.save() stores its state (Adam moments,
    # step count, LR schedule) and a resumed run continues exactly where it stopped.
    if getattr(model, "optimizer", None) is not None:
        print("Restored optimizer state from checkpoint (--learning_rate/--warmup_steps/--clipnorm ignored)")
    else:
        model.compile(optimizer=build_optimizer(learning_rate, warmup_steps, clipnorm))
    optimizer = model.optimizer
    train_step = make_train_step(model, optimizer)
    val_step = make_val_step(model)

    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    for epoch in range(start_epoch, epochs):
        for step, (images, targets) in enumerate(train_dataset):
            global_step = epoch * steps_per_epoch + step
            total_loss, loss_terms = train_step(images, targets)
            _log_scalars(train_writer, loss_terms, total_loss, global_step, "train")
            with train_writer.as_default():
                tf.summary.scalar("train/learning_rate", optimizer.learning_rate, step=global_step)
            if step % 10 == 0:
                terms = ", ".join(f"{k}={v:.4f}" for k, v in loss_terms.items())
                print(f"epoch {epoch} step {step}: loss={total_loss:.4f} ({terms})")

        epoch_end_step = (epoch + 1) * steps_per_epoch - 1

        if val_ids:
            val_losses, val_terms_sum = [], None
            for images, targets in val_dataset:
                total_loss, loss_terms = val_step(images, targets)
                val_losses.append(float(total_loss))
                if val_terms_sum is None:
                    val_terms_sum = {k: [] for k in loss_terms}
                for k, v in loss_terms.items():
                    val_terms_sum[k].append(float(v))
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_terms = {k: sum(v) / len(v) for k, v in val_terms_sum.items()}
            _log_scalars(val_writer, avg_val_terms, avg_val_loss, epoch_end_step, "val")
            print(f"epoch {epoch} val: loss={avg_val_loss:.4f}")

            if map_every and (epoch + 1) % map_every == 0:
                ap_metrics = evaluate(model, data_root, val_ids)
                with val_writer.as_default():
                    for name, value in ap_metrics.items():
                        tf.summary.scalar(f"val/{name}", value, step=epoch_end_step)
                print(f"epoch {epoch} metrics: {ap_metrics}")
                _log_heatmap_sample(val_writer, model, val_ids, data_root, epoch_end_step)

        train_writer.flush()
        val_writer.flush()

        if checkpoint_path:
            model.save(checkpoint_path)
            with open(_epoch_state_path(checkpoint_path), "w") as f:
                f.write(str(epoch + 1))
            print(f"Saved checkpoint to {checkpoint_path} (completed epoch {epoch + 1}/{epochs})")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MobileNet 3D vehicle detector on KITTI.")
    parser.add_argument("--data_root", required=True, help="KITTI root dir containing training/{image_2,label_2,calib}")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate after warmup")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear learning-rate warmup steps")
    parser.add_argument("--clipnorm", type=float, default=10.0, help="Global gradient-norm clip (0 to disable)")
    parser.add_argument("--no_pretrained", action="store_true", help="Skip loading ImageNet backbone weights")
    parser.add_argument("--checkpoint_path", default="mobilenet_3d_detector.keras")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N training samples (for smoke tests)")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of samples held out for validation")
    parser.add_argument(
        "--mapping_path",
        default=None,
        help="Path to KITTI devkit's train_mapping.txt, for a sequence-aware (leakage-free) split",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the train/val split")
    parser.add_argument("--log_dir", default="logs", help="TensorBoard log directory")
    parser.add_argument("--map_every", type=int, default=1, help="Compute AP metrics every N epochs (0 to disable)")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        clipnorm=args.clipnorm,
        pretrained_backbone=not args.no_pretrained,
        checkpoint_path=args.checkpoint_path,
        limit=args.limit,
        val_fraction=args.val_fraction,
        mapping_path=args.mapping_path,
        seed=args.seed,
        log_dir=args.log_dir,
        map_every=args.map_every,
    )
