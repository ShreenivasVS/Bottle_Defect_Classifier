"""
features.py
Extracts a feature vector from a bottle image:
  - HOG (Histogram of Oriented Gradients) on grayscale -> captures shape/edge
    irregularities like cracks and breaks
  - Color histograms on RGB -> captures contamination / discoloration
"""

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray

IMG_SIZE = 256          # images are resized to IMG_SIZE x IMG_SIZE before feature extraction
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
COLOR_BINS = 32


def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img)


def extract_features(path):
    img = load_image(path)
    gray = rgb2gray(img)

    hog_feat = hog(
        gray,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        orientations=9,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    color_feats = []
    for channel in range(3):
        hist, _ = np.histogram(
            img[:, :, channel], bins=COLOR_BINS, range=(0, 255), density=True
        )
        color_feats.append(hist)
    color_feat = np.concatenate(color_feats)

    return np.concatenate([hog_feat, color_feat])


def extract_batch(records, verbose=True):
    X, y = [], []
    for i, rec in enumerate(records):
        X.append(extract_features(rec["path"]))
        y.append(rec["label"])
        if verbose and (i + 1) % 50 == 0:
            print(f"  extracted {i + 1}/{len(records)}")
    return np.array(X), np.array(y)
