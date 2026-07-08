"""Entropy (EN) metric and image loading helper."""

import numpy as np
from PIL import Image


def load_gray(path):
    """Load an image as a grayscale array in [0, 255]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def compute_en(image):
    """Compute entropy."""
    # Use the same 256-bin histogram setting as the original evaluation script.
    histogram, _ = np.histogram(image, bins=256, range=(0, 255))

    # Convert counts to probabilities before applying Shannon entropy.
    histogram = histogram / float(np.sum(histogram))
    return float(-np.sum(histogram * np.log2(histogram + 1e-7)))
