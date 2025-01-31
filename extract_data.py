import zipfile
import os

# Define paths
zip_file = "archive.zip"  # Replace with the actual file name
extract_folder = "dataset"

# Extract if not already extracted
if not os.path.exists(extract_folder):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"✅ Dataset extracted to: {extract_folder}")
else:
    print("⚠️ Dataset already extracted.")
