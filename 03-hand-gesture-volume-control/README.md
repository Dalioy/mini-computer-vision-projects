# Hand Gesture Volume Control

Controls the system volume with a thumb-index **pinch gesture**, using
Google's **MediaPipe HandLandmarker** for hand tracking and a small
cross-platform volume backend.

## How It Works

1. **Hand tracking** — MediaPipe's `HandLandmarker` (running in `VIDEO`
   mode) detects 21 hand landmarks per frame.
2. **Gesture measurement** — the pixel distance between the thumb tip
   (landmark 4) and index-finger tip (landmark 8) is measured every frame.
3. **Mapping** — that distance is linearly mapped from a
   `[MIN_DISTANCE, MAX_DISTANCE]` pixel range to a `0–100%` volume level,
   then smoothed with exponential smoothing to avoid jitter.
4. **System volume** — the smoothed level is applied via
   [`pycaw`](https://github.com/AndreMiras/pycaw) on Windows, `osascript`
   on macOS, or `amixer` on Linux (see `src/volume_control.py`).
5. An on-screen bar shows the current level along with the tracked
   fingertips.

## Project Structure

```
03-hand-gesture-volume-control/
├── models/
│   └── hand_landmarker.task   # MediaPipe hand-landmark model
├── src/
│   ├── gesture_volume_control.py  # Main app
│   └── volume_control.py          # Cross-platform volume backend
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Default: local webcam (index 0)
python src/gesture_volume_control.py

# Custom source (e.g. an IP camera / DroidCam stream)
python src/gesture_volume_control.py --source http://<camera-ip>:4747/video
```

Pinch your thumb and index finger together to lower the volume, and
spread them apart to raise it. Press **`q`** to quit.

## Notes

- System volume control is OS-dependent: fully supported on Windows
  (via `pycaw`) and macOS (via `osascript`); on Linux it relies on
  `amixer`/PulseAudio being available. On unsupported setups, the app
  still runs — the on-screen level just won't change the real system
  volume.
- `MIN_DISTANCE` / `MAX_DISTANCE` in `gesture_volume_control.py` may need
  tuning depending on your camera's resolution and distance from the hand.
