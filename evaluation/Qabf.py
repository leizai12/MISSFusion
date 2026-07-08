"""Qabf edge information transfer metric for image fusion."""

import math

import numpy as np
from scipy.signal import convolve2d


def _conv2_same(kernel, image):
    """MATLAB-like 2D convolution with zero padding."""
    kernel = np.flip(kernel)
    padded = np.pad(image, ((1, 1), (1, 1)), mode="constant", constant_values=(0, 0))
    return convolve2d(padded, kernel, mode="valid")


def _strength_angle(image):
    """Compute Sobel gradient strength and angle."""
    # Sobel kernels follow the original Qabf implementation.
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sx = _conv2_same(h3, image)
    sy = _conv2_same(h1, image)
    strength = np.sqrt(np.multiply(sx, sx) + np.multiply(sy, sy))

    # Use pi/2 when the horizontal response is zero.
    n, m = image.shape
    angle = np.zeros((n, m))
    zero_mask = sx == 0
    angle[~zero_mask] = np.arctan(sy[~zero_mask] / sx[~zero_mask])
    angle[zero_mask] = math.pi / 2
    return strength, angle


def compute_qabf(ir, vi, fused):
    """Compute Qabf for infrared-visible image fusion."""
    # Constants are kept consistent with the standard Qabf formulation.
    tg = 0.9994
    kg = -15
    dg = 0.5
    ta = 0.9879
    ka = -22
    da = 0.8

    grad_ir, angle_ir = _strength_angle(ir)
    grad_vi, angle_vi = _strength_angle(vi)
    grad_fused, angle_fused = _strength_angle(fused)

    def edge_score(src_angle, src_grad, fused_angle, fused_grad):
        # Compare edge strength and orientation between a source image and the fused image.
        mask = src_grad > fused_grad
        with np.errstate(divide="ignore", invalid="ignore"):
            gaf = np.where(mask, fused_grad / src_grad, np.where(src_grad == fused_grad, fused_grad, src_grad / fused_grad))
        aaf = 1 - np.abs(src_angle - fused_angle) / (math.pi / 2)
        qg = tg / (1 + np.exp(kg * (gaf - dg)))
        qa = ta / (1 + np.exp(ka * (aaf - da)))
        return qg * qa

    q_ir = edge_score(angle_ir, grad_ir, angle_fused, grad_fused)
    q_vi = edge_score(angle_vi, grad_vi, angle_fused, grad_fused)

    # Weight edge preservation by source gradient strength.
    denominator = np.sum(grad_ir + grad_vi)
    numerator = np.sum(np.multiply(q_ir, grad_ir) + np.multiply(q_vi, grad_vi))
    return float(numerator / denominator)
