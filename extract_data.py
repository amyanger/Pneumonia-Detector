"""Extract the Kaggle "Chest X-Ray Images (Pneumonia)" zip into the layout the
other scripts expect:

    dataset/chest_xray/train/{NORMAL,PNEUMONIA}/*.jpeg
    dataset/chest_xray/test/{NORMAL,PNEUMONIA}/*.jpeg
    dataset/chest_xray/val/{NORMAL,PNEUMONIA}/*.jpeg

Some versions of the Kaggle zip contain a nested duplicate
(chest_xray/chest_xray/...) and __MACOSX metadata. Only one copy is kept and
__MACOSX is skipped.

Paths are relative to the current working directory, so run every script from
the same directory (normally the repo root).

Usage:
    python extract_data.py [path/to/archive.zip]
"""

import os
import shutil
import sys
import zipfile

DEFAULT_ZIP = "archive.zip"
DATASET_ROOT = "dataset"
DATASET_DIR = os.path.join(DATASET_ROOT, "chest_xray")


def find_split_root(folder):
    """Return the first directory (breadth first) that has train/ and test/ subfolders."""
    queue = [folder]
    while queue:
        current = queue.pop(0)
        entries = sorted(os.listdir(current))
        if "train" in entries and "test" in entries:
            return current
        for name in entries:
            path = os.path.join(current, name)
            if os.path.isdir(path) and name != "__MACOSX" and not name.startswith("."):
                queue.append(path)
    return None


def main():
    zip_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZIP

    if os.path.isdir(os.path.join(DATASET_DIR, "train")):
        print(f"Dataset already extracted at {DATASET_DIR}, nothing to do.")
        return

    if not os.path.isfile(zip_file):
        print(f"Error: zip file not found at '{zip_file}'.")
        print("Download the Kaggle 'Chest X-Ray Images (Pneumonia)' dataset and run:")
        print("   python extract_data.py path/to/archive.zip")
        sys.exit(1)

    # Extract into a staging folder first, then move the split folders into place
    staging = os.path.join(DATASET_ROOT, "_extract_tmp")
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        members = [m for m in zip_ref.namelist() if "__MACOSX" not in m.split("/")]
        zip_ref.extractall(staging, members=members)

    split_root = find_split_root(staging)
    if split_root is None:
        shutil.rmtree(staging, ignore_errors=True)
        print("Error: could not find train/ and test/ folders inside the zip.")
        sys.exit(1)

    os.makedirs(DATASET_DIR, exist_ok=True)
    for split in ["train", "test", "val"]:
        src = os.path.join(split_root, split)
        if os.path.isdir(src):
            shutil.move(src, os.path.join(DATASET_DIR, split))
    shutil.rmtree(staging)

    print(f"Dataset extracted to: {DATASET_DIR}")


if __name__ == "__main__":
    main()
