import tensorflow as tf


def focal_loss(logits, gt, alpha: float = 2.0, beta: float = 4.0):
    # Work from logits: clipping sigmoid(logits) zeroes the gradient once the logits saturate,
    # which permanently kills the heatmap head. log_sigmoid stays stable and keeps a gradient.
    pred = tf.sigmoid(logits)
    pos_mask = tf.cast(tf.equal(gt, 1.0), tf.float32)
    neg_mask = tf.cast(tf.less(gt, 1.0), tf.float32)
    neg_weights = tf.pow(1.0 - gt, beta)

    pos_loss = tf.math.log_sigmoid(logits) * tf.pow(1.0 - pred, alpha) * pos_mask
    neg_loss = tf.math.log_sigmoid(-logits) * tf.pow(pred, alpha) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    return tf.cond(
        num_pos > 0,
        lambda: -(pos_loss + neg_loss) / num_pos,
        lambda: -neg_loss,
    )


def gather_at_indices(feature_map, indices):
    batch_size = tf.shape(feature_map)[0]
    height = tf.shape(feature_map)[1]
    width = tf.shape(feature_map)[2]
    channels = feature_map.shape[-1]

    flat = tf.reshape(feature_map, [batch_size, height * width, channels])
    return tf.gather(flat, indices, batch_dims=1)


def masked_l1_loss(pred_map, indices, mask, target):
    pred = gather_at_indices(pred_map, indices)
    mask = tf.expand_dims(mask, -1)
    loss = tf.reduce_sum(tf.abs(pred - target) * mask)
    return loss / (tf.reduce_sum(mask) + 1e-4)


def compute_losses(predictions, targets, weights=None):
    weights = weights or {"heatmap": 1.0, "offset": 1.0, "depth": 1.0, "dims": 1.0, "orientation": 1.0}

    heatmap_loss = focal_loss(predictions["heatmap"], targets["heatmap"])
    offset_loss = masked_l1_loss(predictions["offset"], targets["indices"], targets["mask"], targets["offset"])
    depth_loss = masked_l1_loss(tf.sigmoid(predictions["depth"]), targets["indices"], targets["mask"], targets["depth"])
    dims_loss = masked_l1_loss(predictions["dims"], targets["indices"], targets["mask"], targets["dims"])
    orientation_loss = masked_l1_loss(
        predictions["orientation"], targets["indices"], targets["mask"], targets["orientation"]
    )

    total = (
        weights["heatmap"] * heatmap_loss
        + weights["offset"] * offset_loss
        + weights["depth"] * depth_loss
        + weights["dims"] * dims_loss
        + weights["orientation"] * orientation_loss
    )

    return total, {
        "heatmap": heatmap_loss,
        "offset": offset_loss,
        "depth": depth_loss,
        "dims": dims_loss,
        "orientation": orientation_loss,
    }
