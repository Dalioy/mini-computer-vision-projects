"""
Hand-gesture volume control.

Tracks the distance between the thumb tip and index-finger tip using
MediaPipe's HandLandmarker, maps that distance to a 0-100 volume level,
and applies it to the system volume in real time.
"""

import argparse
import math
import os
import time

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from volume_control import set_system_volume

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")

THUMB_TIP = 4
INDEX_TIP = 8

# Pinch distance (in pixels, at typical webcam distance) mapped to 0-100% volume.
MIN_DISTANCE = 25
MAX_DISTANCE = 200

SMOOTHING = 0.3  # Exponential smoothing factor for the volume level (0 = none, 1 = instant).


def create_landmarker(model_path: str) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def distance_to_percent(distance: float) -> float:
    clamped = max(MIN_DISTANCE, min(MAX_DISTANCE, distance))
    return (clamped - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE) * 100


def draw_ui(frame, thumb_pt, index_pt, volume_pct: float) -> None:
    cv2.circle(frame, thumb_pt, 8, (255, 0, 0), -1)
    cv2.circle(frame, index_pt, 8, (255, 0, 0), -1)
    cv2.line(frame, thumb_pt, index_pt, (255, 0, 0), 2)

    bar_x, bar_top, bar_bottom = 40, 60, 300
    bar_fill_y = int(bar_bottom - (volume_pct / 100) * (bar_bottom - bar_top))
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 30, bar_bottom), (0, 255, 0), 2)
    cv2.rectangle(frame, (bar_x, bar_fill_y), (bar_x + 30, bar_bottom), (0, 255, 0), -1)
    cv2.putText(frame, f"{int(volume_pct)}%", (bar_x - 10, bar_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def run(source: str, model_path: str) -> None:
    landmarker = create_landmarker(model_path)

    capture_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    print("[INFO] Press 'q' to quit.")
    smoothed_volume = 0.0
    start_time = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame, stopping.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            h, w = frame.shape[:2]
            thumb_pt = (int(hand[THUMB_TIP].x * w), int(hand[THUMB_TIP].y * h))
            index_pt = (int(hand[INDEX_TIP].x * w), int(hand[INDEX_TIP].y * h))

            pixel_distance = math.hypot(index_pt[0] - thumb_pt[0], index_pt[1] - thumb_pt[1])
            target_volume = distance_to_percent(pixel_distance)
            smoothed_volume += SMOOTHING * (target_volume - smoothed_volume)

            set_system_volume(smoothed_volume)
            draw_ui(frame, thumb_pt, index_pt, smoothed_volume)

        cv2.imshow("Hand Gesture Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control system volume with a thumb-index pinch gesture.")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. '0') or a stream URL (e.g. an IP-camera address).",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the hand_landmarker.task model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source, args.model_path)
