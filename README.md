# Computer Vision Projects

This repository is a **reconstruction** of three previously separate
computer vision projects, consolidated into a single monorepo — one
folder per project, each with its own source code, dependencies, and
README.

All three projects are built around **classical feature detection and
matching** (ORB/SIFT + homography estimation) and real-time video
processing with OpenCV.

## Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | [ORB Feature Matching](./01-orb-feature-matching) | Real-time object recognition: matches a live camera feed against a small reference-image database using ORB descriptors. |
| 2 | [Panorama Stitching](./02-panorama-stitching) | Stitches a sequence of overlapping images into a panorama using ORB matching, RANSAC homography, and perspective warping — with memory-safe sequential processing for large image sets. |
| 3 | [Hand Gesture Volume Control](./03-hand-gesture-volume-control) | Controls the system volume with a thumb-index pinch gesture, using MediaPipe hand tracking and a cross-platform volume backend. |

Each project folder is self-contained: install its `requirements.txt` and
run its script(s) independently — see the linked README for details.

## Tech Stack

- **OpenCV** — feature detection (ORB/SIFT), matching, homography, video I/O
- **NumPy** — array/geometry operations
- **MediaPipe** — hand landmark tracking
- **pycaw** — Windows system volume control
