"""Evaluate the trained model on the held-out test set.

Loads backend/pneumonia_detector_model.keras (resolved relative to this script,
or the MODEL_PATH environment variable if set) and dataset/chest_xray/test
(relative to the current working directory, as produced by extract_data.py).
Images go through the same preprocessing as training via
preprocess_data.load_images.

Usage:
    python evaluate_model.py [--threshold 0.5]
"""

import argparse
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow import keras

from preprocess_data import CATEGORIES, DATASET_DIR, load_images

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "backend", "pneumonia_detector_model.keras")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
TEST_PATH = os.path.join(DATASET_DIR, "test")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the pneumonia classifier.")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability above which an image is classed as PNEUMONIA (default 0.5)",
    )
    args = parser.parse_args()

    model = keras.models.load_model(MODEL_PATH)
    X_test, y_test, _ = load_images(TEST_PATH)
    print(f"Loaded {len(y_test)} test images from {TEST_PATH}")

    probs = model.predict(X_test, batch_size=32).ravel()
    preds = (probs > args.threshold).astype(int)

    print(f"\nThreshold: {args.threshold}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, labels=[0, 1], target_names=CATEGORIES, zero_division=0))

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    print("Confusion matrix (rows = true, columns = predicted):")
    print(f"{'':>12}{'NORMAL':>10}{'PNEUMONIA':>11}")
    for name, row in zip(CATEGORIES, cm):
        print(f"{name:>12}{row[0]:>10}{row[1]:>11}")

    if len(np.unique(y_test)) == 2:
        print(f"\nROC AUC: {roc_auc_score(y_test, probs):.4f}")
    else:
        print("\nROC AUC: not defined (test set has only one class)")


if __name__ == "__main__":
    main()
