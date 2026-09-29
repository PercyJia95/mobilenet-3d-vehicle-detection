import tensorflow as tf

from .backbone import build_mobilenet_backbone
from .fpn import build_fpn

PYRAMID_NAMES = ["P2", "P3", "P4", "P5", "P6"]


def build_mobilenet_fpn(
    input_shape=(512, 512, 3),
    out_channels: int = 256,
    pretrained_backbone: bool = True,
    extra_level: bool = True,
) -> tf.keras.Model:
    backbone = build_mobilenet_backbone(input_shape=input_shape, pretrained=pretrained_backbone)
    pyramid = build_fpn(backbone.outputs, out_channels=out_channels, extra_level=extra_level)

    names = PYRAMID_NAMES if extra_level else PYRAMID_NAMES[:-1]
    outputs = dict(zip(names, pyramid))

    return tf.keras.Model(inputs=backbone.input, outputs=outputs, name="mobilenet_fpn")
