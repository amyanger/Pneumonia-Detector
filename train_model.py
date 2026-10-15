"""Train the pneumonia classifier on the arrays saved by preprocess_data.py.

Reads X_train.npy, X_val.npy, y_train.npy, y_val.npy from the current working
directory. Saves the model to backend/pneumonia_detector_model.keras, resolved
relative to this script so the backend picks it up. Set the MODEL_PATH
environment variable to save somewhere else.

Usage:
    python train_model.py [--epochs 20]
"""

import argparse
import os

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "backend", "pneumonia_detector_model.keras")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)


def build_model():
    # Pretrained MobileNetV2 with frozen weights
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(224, 224, 1)),
            layers.Conv2D(3, (3, 3), padding="same"),  # Grayscale to 3 channels
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),  # 0 = Normal, 1 = Pneumonia
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the pneumonia classifier.")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    X_train = np.load("X_train.npy")
    X_val = np.load("X_val.npy")
    y_train = np.load("y_train.npy")
    y_val = np.load("y_val.npy")

    # Class weights to offset the NORMAL/PNEUMONIA imbalance
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    model = build_model()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model training complete. Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
