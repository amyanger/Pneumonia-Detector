"""Show one random training image per class.

Expects the layout produced by extract_data.py (dataset/chest_xray/train),
relative to the current working directory.
"""

import os
import random

import cv2
import matplotlib.pyplot as plt

DATASET_PATH = "dataset/chest_xray/train"
CATEGORIES = ["NORMAL", "PNEUMONIA"]


def show_sample_images():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for i, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_PATH, category)
        if not os.path.isdir(folder):
            print(f"Warning: folder '{folder}' is missing.")
            continue

        # Skip hidden files such as .DS_Store
        names = [n for n in os.listdir(folder) if not n.startswith(".")]
        random.shuffle(names)

        img = None
        skipped = 0
        for name in names:
            img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                break
            skipped += 1

        if skipped:
            print(f"Skipped {skipped} unreadable file(s) in '{folder}'.")
        if img is None:
            print(f"Warning: no readable images in '{folder}'.")
            continue

        img = cv2.resize(img, (224, 224))
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(category)
        axes[i].axis("off")

    plt.show()


def main():
    show_sample_images()


if __name__ == "__main__":
    main()
