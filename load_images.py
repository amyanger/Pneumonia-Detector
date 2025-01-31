import os
import matplotlib.pyplot as plt
import cv2
import random

# Correct dataset path
DATASET_PATH = "dataset/chest_xray/train"
CATEGORIES = ["NORMAL", "PNEUMONIA"]


def show_sample_images():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for i, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_PATH, category)

        # Check if the folder exists and contains files
        if not os.path.exists(folder) or not os.listdir(folder):
            print(f"⚠️ Warning: Folder '{folder}' is empty or missing!")
            continue

        image_name = random.choice(os.listdir(folder))
        image_path = os.path.join(folder, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(category)
        axes[i].axis("off")

    plt.show()


# Run function
show_sample_images()
