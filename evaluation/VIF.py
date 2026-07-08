"""Visual Information Fidelity (VIF) metric for image fusion."""

import numpy as np
from scipy.signal import convolve2d


def _gaussian_kernel(shape, sigma):
    """Create a 2D Gaussian kernel."""
    m, n = [(size - 1.0) / 2.0 for size in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    kernel_sum = kernel.sum()
    if kernel_sum != 0:
        kernel /= kernel_sum
    return kernel


def _vifp_mscale(ref, dist):
    """Compute multi-scale VIF for one reference image."""
    sigma_nsq = 2
    num = 0
    den = 0
    ref = np.asarray(ref, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    # Evaluate information fidelity over four scales.
    for scale in range(1, 5):
        window_size = 2 ** (4 - scale + 1) + 1
        window = _gaussian_kernel((window_size, window_size), window_size / 5)

        # Downsample after the first scale.
        if scale > 1:
            ref = convolve2d(ref, window, mode="valid")
            dist = convolve2d(dist, window, mode="valid")
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, window, mode="valid")
        mu2 = convolve2d(dist, window, mode="valid")
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, window, mode="valid") - mu1_sq
        sigma2_sq = convolve2d(dist * dist, window, mode="valid") - mu2_sq
        sigma12 = convolve2d(ref * dist, window, mode="valid") - mu1_mu2

        # Numerical guards follow the original VIF implementation.
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0
        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0
        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0
        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    return float(num / den)


def compute_vif(ir, vi, fused):
    """Compute VIF between the fused image and both sources."""
    # IVIF evaluation sums the fidelity with respect to both modalities.
    return float(_vifp_mscale(ir, fused) + _vifp_mscale(vi, fused))
