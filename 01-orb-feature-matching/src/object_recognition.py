"""
ORB-based real-time object recognition.

Builds a small "reference database" of ORB descriptors from a folder of
training images, then matches every incoming camera frame against that
database using a brute-force Hamming matcher + Lowe's ratio test.
"""

import argparse
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "train_images")


def load_reference_images(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load every image in `data_dir` as grayscale, paired with its label."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")

    images, labels = [], []
    for file_name in sorted(os.listdir(data_dir)):
        img = cv2.imread(os.path.join(data_dir, file_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels.append(os.path.splitext(file_name)[0])

    print(f"[INFO] Loaded {len(images)} reference image(s): {labels}")
    return images, labels


def compute_descriptors(images: List[np.ndarray], orb: cv2.ORB) -> List[Optional[np.ndarray]]:
    """Precompute ORB descriptors for every reference image."""
    return [orb.detectAndCompute(img, None)[1] for img in images]


def match_frame(
    frame_gray: np.ndarray,
    reference_descriptors: List[Optional[np.ndarray]],
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
    ratio_thresh: float = 0.75,
    min_good_matches: int = 15,
) -> int:
    """
    Return the index of the best-matching reference image for `frame_gray`,
    or -1 if nothing scores above `min_good_matches`.
    """
    _, frame_des = orb.detectAndCompute(frame_gray, None)
    if frame_des is None:
        return -1

    match_counts = []
    for ref_des in reference_descriptors:
        if ref_des is None:
            match_counts.append(0)
            continue

        matches = matcher.knnMatch(ref_des, frame_des, k=2)
        good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        match_counts.append(len(good))

    best_score = max(match_counts) if match_counts else 0
    return match_counts.index(best_score) if best_score > min_good_matches else -1


def run(source: str, data_dir: str, match_threshold: int) -> None:
    orb = cv2.ORB_create()
    images, labels = load_reference_images(data_dir)
    reference_descriptors = compute_descriptors(images, orb)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # `source` may be a webcam index ("0") or a stream URL.
    capture_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    print("[INFO] Press 'q' to quit.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame, stopping.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        match_idx = match_frame(frame_gray, reference_descriptors, orb, matcher, min_good_matches=match_threshold)

        if match_idx != -1:
            cv2.putText(frame, labels[match_idx], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("ORB Object Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time object recognition using ORB feature matching.")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. '0') or a stream URL (e.g. an IP-camera address).",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Folder containing the reference/training images.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="Minimum number of good matches required to accept a recognition.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source, args.data_dir, args.threshold)
