# ORB Feature Detection & Matching — Real-Time Object Recognition

Real-time object recognition using **ORB (Oriented FAST and Rotated BRIEF)**
feature detection and brute-force Hamming matching. The program builds a
small reference database from a folder of images, then identifies which
reference object (if any) appears in each incoming camera frame.

## How It Works

1. **Reference database** — every image in `data/train_images/` is loaded
   and its ORB keypoints/descriptors are precomputed once at startup.
2. **Live matching** — for each camera frame, ORB descriptors are extracted
   and compared against every reference descriptor set using a
   `BFMatcher` (Hamming distance).
3. **Lowe's ratio test** filters out ambiguous matches, keeping only
   matches where the best candidate is clearly better than the
   second-best.
4. The reference image with the highest number of good matches is
   accepted as a recognition, provided it clears a minimum match count.

## Project Structure

```
01-orb-feature-matching/
├── data/
│   └── train_images/      # Reference images (add your own here)
├── src/
│   └── object_recognition.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Default: local webcam (index 0)
python src/object_recognition.py

# Custom source (e.g. an IP camera / DroidCam stream)
python src/object_recognition.py --source http://<camera-ip>:4747/video

# Custom training data folder and match sensitivity
python src/object_recognition.py --data-dir data/train_images --threshold 20
```

Press **`q`** to close the video window.

## Notes

- Add more images to `data/train_images/` to recognize new objects — no
  retraining or model files needed, ORB descriptors are computed on the fly.
- Recognition quality depends on the reference images having enough
  distinctive texture/detail; flat or low-texture objects are harder for
  ORB to match reliably.
