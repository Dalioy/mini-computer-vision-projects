"""
Download the drone-capture image-stitching dataset from Kaggle and move it
into `sample_data/raw/`.

Requires a configured Kaggle API token (see:
https://github.com/Kaggle/kaggle-api#api-credentials).
"""

import os
import shutil

import kagglehub

DATASET = "phsophea101/image-stitching-from-drone-capture-opencv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESTINATION_DIR = os.path.join(BASE_DIR, "..", "sample_data", "raw")


def unique_destination(destination_dir: str, file_name: str) -> str:
    """Return a non-colliding path for `file_name` inside `destination_dir`."""
    target_path = os.path.join(destination_dir, file_name)
    base, ext = os.path.splitext(file_name)
    counter = 1
    while os.path.exists(target_path):
        target_path = os.path.join(destination_dir, f"{base}_{counter}{ext}")
        counter += 1
    return target_path


def download_and_move(destination_dir: str = DESTINATION_DIR) -> None:
    cache_path = kagglehub.dataset_download(DATASET)
    print(f"[INFO] Downloaded to cache: {cache_path}")

    os.makedirs(destination_dir, exist_ok=True)
    for file_name in os.listdir(cache_path):
        source_path = os.path.join(cache_path, file_name)
        target_path = unique_destination(destination_dir, file_name)
        shutil.move(source_path, target_path)

    print(f"[DONE] Dataset files moved to: {destination_dir}")


if __name__ == "__main__":
    download_and_move()
