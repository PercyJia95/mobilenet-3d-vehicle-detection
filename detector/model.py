import tensorflow as tf

from mobilenet_fpn import build_mobilenet_fpn

from .config import FPN_OUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES
from .head import build_detection_head


def build_3d_detector(
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    num_classes: int = NUM_CLASSES,
    out_channels: int = FPN_OUT_CHANNELS,
    pretrained_backbone: bool = True,
) -> tf.keras.Model:
    fpn = build_mobilenet_fpn(
        input_shape=input_shape,
        out_channels=out_channels,
        pretrained_backbone=pretrained_backbone,
        extra_level=False,
    )
    # With batches of a few images, batch statistics are too noisy: keep the pretrained BN layers
    # frozen (Keras runs non-trainable BatchNormalization in inference mode).
    for layer in fpn.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")
    # The data pipeline yields pixels in [0, 1]; MobileNetV2's ImageNet weights expect [-1, 1].
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(inputs)
    pyramid = fpn(x)
    heads = build_detection_head(pyramid["P2"], num_classes=num_classes)

    return tf.keras.Model(inputs=inputs, outputs=heads, name="mobilenet_3d_detector")
