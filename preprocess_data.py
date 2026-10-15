"""Load the training images, split them by patient, and save NumPy arrays.

Reads dataset/chest_xray/train (produced by extract_data.py) and writes
X_train.npy, X_val.npy, y_train.npy, y_val.npy. All paths are relative to the
current working directory.

Preprocessing must match the backend and the saved model: grayscale, resized to
224x224, float32 scaled to [0, 1], shape (N, 224, 224, 1). Labels are
0 = NORMAL, 1 = PNEUMONIA.

The Kaggle data has several images per patient, so the train/validation split
is grouped by patient to keep the same person out of both sets.
"""

import os
import re

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

DATASET_DIR = "dataset/chest_xray"
TRAIN_PATH = os.path.join(DATASET_DIR, "train")
CATEGORIES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224


def patient_id(filename):
    """Patient key from a Kaggle filename.

    PNEUMONIA: person1000_virus_1681.jpeg -> person1000
    NORMAL:    IM-0115-0001.jpeg / NORMAL2-IM-0927-0001.jpeg -> IM-0115 / IM-0927
    Falls back to the file stem if the name does not match either pattern.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"(person\d+)_", name)
    if match:
        return match.group(1)
    if name.startswith("NORMAL2-"):
        name = name[len("NORMAL2-"):]
    match = re.match(r"(IM-\d+)", name)
    if match:
        return match.group(1)
    return name


def load_images(dataset_path=TRAIN_PATH):
    """Load all images under dataset_path/{NORMAL,PNEUMONIA}.

    Returns (data, labels, filenames). Hidden files and anything OpenCV cannot
    read are skipped.
    """
    data = []
    labels = []
    filenames = []
    skipped = 0

    for label, category in enumerate(CATEGORIES):
        folder_path = os.path.join(dataset_path, category)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Missing folder '{folder_path}'. Run extract_data.py first."
            )

        for img_name in sorted(os.listdir(folder_path)):
            if img_name.startswith("."):
                skipped += 1
                continue
            img = cv2.imread(os.path.join(folder_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue

            data.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            labels.append(label)
            filenames.append(img_name)

    if skipped:
        print(f"Skipped {skipped} hidden or unreadable file(s) in {dataset_path}")

    # Convert from uint8 straight to float32 to avoid a float64 copy
    data = np.asarray(data, dtype=np.uint8).astype(np.float32) / 255.0
    data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.asarray(labels, dtype=np.int64)

    return data, labels, filenames


def patient_split(labels, groups, test_size=0.2, random_state=42):
    """Split indices by patient, separately per class so class ratios stay similar."""
    train_idx = []
    val_idx = []
    for label in np.unique(labels):
        class_idx = np.where(labels == label)[0]
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        tr, va = next(splitter.split(class_idx, groups=groups[class_idx]))
        train_idx.extend(class_idx[tr])
        val_idx.extend(class_idx[va])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def main():
    X, y, filenames = load_images()
    groups = np.array([patient_id(f) for f in filenames])

    train_idx, val_idx = patient_split(y, groups)
    overlap = set(groups[train_idx]) & set(groups[val_idx])
    assert not overlap, f"Patients found in both splits: {sorted(overlap)[:10]}"

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    np.save("X_train.npy", X_train)
    np.save("X_val.npy", X_val)
    np.save("y_train.npy", y_train)
    np.save("y_val.npy", y_val)

    for name, labels, idx in [("Train", y_train, train_idx), ("Validation", y_val, val_idx)]:
        print(
            f"{name}: {len(labels)} images from {len(set(groups[idx]))} patients "
            f"(NORMAL={int(np.sum(labels == 0))}, PNEUMONIA={int(np.sum(labels == 1))})"
        )
    print("Preprocessing complete, no patient appears in both splits.")


if __name__ == "__main__":
    main()
