import numpy as np
from PIL import Image


def load_gray(path):
    """Load an image as a grayscale array in [0, 255]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def compute_en(image):
    """Compute entropy."""
    histogram, _ = np.histogram(image, bins=256, range=(0, 255))
    histogram = histogram / float(np.sum(histogram))
    return float(-np.sum(histogram * np.log2(histogram + 1e-7)))
