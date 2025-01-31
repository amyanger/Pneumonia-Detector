import numpy as np
import tensorflow as tf
from tensorflow import keras  # ✅ Fix for Pylance error
from tensorflow.keras import layers  # type: ignore
from sklearn.utils.class_weight import compute_class_weight

# ✅ Load preprocessed data
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")

# ✅ Compute class weights (Fixes dataset imbalance)
class_weights = compute_class_weight(
    "balanced", classes=np.array([0, 1]), y=y_train
)  # ✅ Fix: Convert to NumPy array
class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1],
}  # ✅ Convert to dictionary

# ✅ Data Augmentation (Prevents Overfitting)
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# ✅ Pretrained MobileNetV2 Model (Faster & More Accurate)
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # Freeze pre-trained layers

# ✅ Define Model Architecture
model = keras.Sequential(
    [
        keras.layers.Input(shape=(224, 224, 1)),  # Fixes `input_shape` warning
        layers.Conv2D(3, (3, 3), padding="same"),  # Convert grayscale to 3 channels
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(
            1, activation="sigmoid"
        ),  # Binary classification (0=Normal, 1=Pneumonia)
    ]
)

# ✅ Compile Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Learning Rate Scheduler (Adaptive Learning)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)

# ✅ Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=20,  # Increased from 10 to 20
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,  # Apply class balancing
    callbacks=[lr_scheduler],  # Apply learning rate scheduler
)

# ✅ Save the trained model
model.save("pneumonia_detector_model.keras")  # Updated to .keras format

print("🎯 Model training complete! Saved as pneumonia_detector_model.keras")
