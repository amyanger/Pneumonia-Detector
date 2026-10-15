"""Predict pneumonia for a single chest X-ray and show the image.

Loads backend/pneumonia_detector_model.keras, resolved relative to this script
(or the MODEL_PATH environment variable if set). The image path is taken as
given, relative to the current working directory.

Usage:
    python test_model.py dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "backend", "pneumonia_detector_model.keras")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: image file not found at {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: OpenCV could not read the image file at {image_path}")
        sys.exit(1)

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0  # Scale to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Channel dimension
    img = np.expand_dims(img, axis=0)  # Batch dimension
    return img


def predict_pneumonia(model, image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        label = f"Pneumonia Detected (Confidence: {prediction:.2f})"
    else:
        label = f"Normal Lung (Confidence: {1 - prediction:.2f})"

    img_display = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_display, cmap="gray")
    plt.title(label)
    plt.axis("off")
    plt.show()

    print(label)


def main():
    if len(sys.argv) < 2:
        print("Please provide an image path. Example:")
        print("   python test_model.py dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    predict_pneumonia(model, sys.argv[1])


if __name__ == "__main__":
    main()
