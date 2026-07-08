"""Spatial Frequency (SF) metric for fused images."""

import numpy as np


def compute_sf(image):
    """Compute spatial frequency."""
    image = np.asarray(image)

    # Row and column differences measure horizontal and vertical activity.
    rf = np.diff(image, axis=0)
    cf = np.diff(image, axis=1)

    # Combine row and column frequencies into the final SF score.
    rf_value = np.sqrt(np.mean(np.mean(rf ** 2)))
    cf_value = np.sqrt(np.mean(np.mean(cf ** 2)))
    return float(np.sqrt(rf_value ** 2 + cf_value ** 2))
