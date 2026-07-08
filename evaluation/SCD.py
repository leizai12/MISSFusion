"""Sum of Correlations of Differences (SCD) metric."""

import numpy as np


def _corr2(a, b):
    """Compute 2D correlation coefficient."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def compute_scd(ir, vi, fused):
    """Compute sum of correlations of differences."""
    # Measure how each source modality is preserved in the fusion residuals.
    return float(_corr2(fused - vi, ir) + _corr2(fused - ir, vi))
