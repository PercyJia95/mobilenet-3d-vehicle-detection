import tensorflow as tf


def _conv_head(x, out_channels, name, bias_init=None):
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name=f"{name}_conv")(x)
    bias_initializer = "zeros" if bias_init is None else tf.keras.initializers.Constant(bias_init)
    return tf.keras.layers.Conv2D(out_channels, 1, name=name, bias_initializer=bias_initializer)(x)


def build_detection_head(feature, num_classes: int = 1):
    heatmap = _conv_head(feature, num_classes, "heatmap", bias_init=-2.19)
    offset = _conv_head(feature, 2, "offset")
    depth = _conv_head(feature, 1, "depth")
    dims = _conv_head(feature, 3, "dims")
    orientation = _conv_head(feature, 2, "orientation")

    return {
        "heatmap": heatmap,
        "offset": offset,
        "depth": depth,
        "dims": dims,
        "orientation": orientation,
    }
