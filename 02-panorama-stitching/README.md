# Panorama / Image Stitching

Stitches a sequence of overlapping images (e.g. drone-capture frames) into
a single panorama using **ORB feature matching**, **RANSAC homography
estimation**, and **perspective warping**.

## How It Works

1. Images are processed **sequentially** rather than all at once — each
   new image is stitched onto the panorama built so far — to keep memory
   usage low on large datasets.
2. For every pair, ORB keypoints/descriptors are extracted and matched
   with a brute-force Hamming matcher, filtered with Lowe's ratio test.
3. A homography is estimated with `cv2.findHomography` + RANSAC to
   reject outlier matches.
4. The new image is warped into the growing panorama's coordinate space
   and composited onto a shared canvas.
5. A memory guard stops the process early (saving progress) if available
   RAM drops below a threshold, and large input images are automatically
   downscaled before matching.

## Project Structure

```
02-panorama-stitching/
├── sample_data/
│   └── images/           # A small example sequence to stitch
├── src/
│   ├── stitcher.py        # Main stitching pipeline
│   └── download_dataset.py  # Optional: pull a larger dataset from Kaggle
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Stitch the bundled sample images
python src/stitcher.py --input-dir sample_data/images --output-dir output

# Stitch your own sequence
python src/stitcher.py --input-dir path/to/frames --output-dir path/to/output --min-matches 15
```

The result is saved as `output/panorama.jpg`.

### Optional: download a larger dataset

```bash
python src/download_dataset.py
```

Downloads the [drone-capture image-stitching dataset](https://www.kaggle.com/datasets/phsophea101/image-stitching-from-drone-capture-opencv)
from Kaggle into `sample_data/raw/` (requires a configured Kaggle API
token).

## Notes

- Input images should be named/sorted so that consecutive files overlap
  (e.g. sequential frames from a drone flight or a panning camera).
- If a pair doesn't have enough good matches (`--min-matches`), that
  image is skipped rather than corrupting the panorama.
