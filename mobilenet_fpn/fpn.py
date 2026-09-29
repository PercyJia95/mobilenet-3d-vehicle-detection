import tensorflow as tf


def build_fpn(features, out_channels: int = 256, extra_level: bool = True):
    laterals = [tf.keras.layers.Conv2D(out_channels, 1, name=f"fpn_lateral_{i}")(x) for i, x in enumerate(features)]

    for i in range(len(laterals) - 1, 0, -1):
        top_down = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name=f"fpn_upsample_{i}")(
            laterals[i]
        )
        laterals[i - 1] = tf.keras.layers.Add(name=f"fpn_add_{i - 1}")([laterals[i - 1], top_down])

    outputs = [
        tf.keras.layers.Conv2D(out_channels, 3, padding="same", name=f"fpn_output_{i}")(x)
        for i, x in enumerate(laterals)
    ]

    if extra_level:
        outputs.append(tf.keras.layers.MaxPool2D(pool_size=1, strides=2, name="fpn_extra_p6")(outputs[-1]))

    return outputs
