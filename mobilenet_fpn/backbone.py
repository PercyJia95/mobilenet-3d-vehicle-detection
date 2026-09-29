import tensorflow as tf

STAGE_LAYERS = {
    "C2": "block_3_expand_relu",
    "C3": "block_6_expand_relu",
    "C4": "block_13_expand_relu",
    "C5": "out_relu",
}


def build_mobilenet_backbone(input_shape=(512, 512, 3), pretrained: bool = True) -> tf.keras.Model:
    weights = "imagenet" if pretrained else None
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)
    outputs = [base.get_layer(name).output for name in STAGE_LAYERS.values()]
    return tf.keras.Model(inputs=base.input, outputs=outputs, name="mobilenetv2_backbone")
