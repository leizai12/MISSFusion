"""Standard Deviation (SD) metric for fused images."""

import numpy as np


def compute_sd(image):
    """Compute standard deviation."""
    image = np.asarray(image)
    m, n = image.shape
    mean_value = np.mean(image)

    # Keep the explicit formula for consistency with the original metric script.
    return float(np.sqrt(np.sum(np.sum((image - mean_value) ** 2)) / (m * n)))
