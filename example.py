import numpy as np

from mobilenet_fpn import build_mobilenet_fpn

if __name__ == "__main__":
    model = build_mobilenet_fpn(input_shape=(512, 512, 3), out_channels=256, pretrained_backbone=True)

    x = np.random.randn(1, 512, 512, 3).astype("float32")
    pyramid = model(x, training=False)

    for name, feat in pyramid.items():
        print(f"{name}: {tuple(feat.shape)}")

    model.save("mobilenet_fpn.keras")
    print("Saved model to mobilenet_fpn.keras")
