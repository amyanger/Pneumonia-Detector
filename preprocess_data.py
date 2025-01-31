import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ✅ Define paths
DATASET_PATH = "dataset/chest_xray/train"
CATEGORIES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224  # Resize all images to 224x224 for consistency


# ✅ Function to load and preprocess images
def load_images():
    data = []
    labels = []

    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_PATH, category)
        label = 0 if category == "NORMAL" else 1  # Assign 0 to NORMAL, 1 to PNEUMONIA

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize

            data.append(img)
            labels.append(label)

    # Convert lists to NumPy arrays and normalize pixel values
    data = np.array(data) / 255.0  # Normalize pixel values (0-1)
    labels = np.array(labels)

    # Reshape images to match TensorFlow input format
    data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return data, labels


# ✅ Load & split dataset
X, y = load_images()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Save the preprocessed data
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)

print(
    f"✅ Preprocessing complete: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples"
)
