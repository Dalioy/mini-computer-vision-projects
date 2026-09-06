"""
Sequential panorama stitching from a directory of images.

Images are stitched one at a time (rather than all at once) to keep
memory usage low, which matters for large drone-capture datasets. Each
pair is matched with ORB + a brute-force Hamming matcher, aligned with a
RANSAC-estimated homography, and warped into a shared canvas.
"""

import argparse
import gc
import os
from typing import Optional

import cv2
import numpy as np
import psutil

MAX_WIDTH = 1200  # Images wider than this are downscaled before matching.


def print_memory_status() -> None:
    mem = psutil.virtual_memory()
    print(f"[MEM] Available: {mem.available / (1024 ** 3):.2f} GB ({mem.available * 100 / mem.total:.1f}%)")


def resize_if_large(img: np.ndarray, max_width: int = MAX_WIDTH) -> np.ndarray:
    """Downscale `img` so its width does not exceed `max_width`."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def warp_pair(base_img: np.ndarray, homography: np.ndarray, warp_img: np.ndarray) -> np.ndarray:
    """Warp `warp_img` by `homography` onto a canvas large enough to also hold `base_img`."""
    h1, w1 = base_img.shape[:2]
    h2, w2 = warp_img.shape[:2]

    corners_base = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_warp = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_warp, homography)

    all_corners = np.concatenate((corners_base, transformed_corners), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    canvas = cv2.warpPerspective(warp_img, translation.dot(homography), (x_max - x_min, y_max - y_min))
    canvas[-y_min:h1 - y_min, -x_min:w1 - x_min] = base_img
    return canvas


def stitch_pair(base_img: np.ndarray, next_img: np.ndarray, min_matches: int = 10) -> Optional[np.ndarray]:
    """
    Match `next_img` against `base_img` with ORB and stitch them together.
    Returns None if too few good matches are found to estimate a homography.
    """
    base_img = resize_if_large(base_img)
    next_img = resize_if_large(next_img)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(base_img, None)
    kp2, des2 = orb.detectAndCompute(next_img, None)
    if des1 is None or des2 is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < min_matches:
        print(f"[WARN] Not enough matches ({len(good)}/{min_matches}), skipping this image.")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None:
        return None

    return warp_pair(next_img, homography, base_img)


def stitch_directory(input_dir: str, output_dir: str, min_matches: int = 10, memory_guard_pct: float = 15.0) -> None:
    """Stitch every image in `input_dir` (sorted by filename) into a single panorama."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    file_list = sorted(f for f in os.listdir(input_dir) if not f.startswith("."))
    if not file_list:
        print("[INFO] No images to process.")
        return

    os.makedirs(output_dir, exist_ok=True)
    panorama = resize_if_large(cv2.imread(os.path.join(input_dir, file_list[0])))

    for i, fname in enumerate(file_list[1:], start=1):
        print(f"[INFO] Stitching image {i + 1}/{len(file_list)}: {fname}")
        next_img = cv2.imread(os.path.join(input_dir, fname))

        result = stitch_pair(panorama, next_img, min_matches=min_matches)
        if result is not None:
            panorama = result

        del next_img
        gc.collect()
        print_memory_status()

        if psutil.virtual_memory().available * 100 / psutil.virtual_memory().total < memory_guard_pct:
            print("[WARN] Low memory — stopping early and saving progress.")
            break

    out_path = os.path.join(output_dir, "panorama.jpg")
    cv2.imwrite(out_path, panorama)
    print(f"[DONE] Panorama saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch a directory of images into a panorama.")
    parser.add_argument("--input-dir", default="sample_data/images", help="Directory of source images.")
    parser.add_argument("--output-dir", default="output", help="Directory to save the stitched panorama.")
    parser.add_argument("--min-matches", type=int, default=10, help="Minimum good matches to accept a pair.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stitch_directory(args.input_dir, args.output_dir, min_matches=args.min_matches)
