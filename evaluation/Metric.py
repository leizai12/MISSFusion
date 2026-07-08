import math
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

try:
    from .Qabf import get_Qabf
except ImportError:
    from Qabf import get_Qabf

METRIC_NAMES = ("EN", "SF", "AG", "SD", "SCD", "VIF", "Qabf", "LPIPS")


def load_gray(path):
    """Load an image as a grayscale array in [0, 255]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def EN_function(image_array):
    """Entropy of the fused image."""
    histogram, _ = np.histogram(image_array, bins=256, range=(0, 255))
    histogram = histogram / float(np.sum(histogram))
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return float(entropy)


def SF_function(image):
    """Spatial frequency of the fused image."""
    image_array = np.array(image)
    rf = np.diff(image_array, axis=0)
    rf_value = np.sqrt(np.mean(np.mean(rf ** 2)))
    cf = np.diff(image_array, axis=1)
    cf_value = np.sqrt(np.mean(np.mean(cf ** 2)))
    return float(np.sqrt(rf_value ** 2 + cf_value ** 2))


def AG_function(image):
    """Average gradient of the fused image."""
    width = image.shape[1] - 1
    height = image.shape[0] - 1
    grad_y, grad_x = np.gradient(image)
    gradient = np.sqrt((np.square(grad_x) + np.square(grad_y)) / 2)
    return float(np.sum(np.sum(gradient)) / (width * height))


def SD_function(image_array):
    """Standard deviation of the fused image."""
    m, n = image_array.shape
    mean_value = np.mean(image_array)
    return float(np.sqrt(np.sum(np.sum((image_array - mean_value) ** 2)) / (m * n)))


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
    """Multi-scale visual information fidelity for one reference image."""
    sigma_nsq = 2
    num = 0
    den = 0
    ref = np.asarray(ref, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    for scale in range(1, 5):
        window_size = 2 ** (4 - scale + 1) + 1
        window = _gaussian_kernel((window_size, window_size), window_size / 5)

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


def VIF_function(ir, vi, fused):
    """Visual information fidelity between the fused image and both sources."""
    return float(_vifp_mscale(ir, fused) + _vifp_mscale(vi, fused))


def _corr2(a, b):
    """2D correlation coefficient."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def SCD_function(ir, vi, fused):
    """Sum of correlations of differences."""
    return float(_corr2(fused - vi, ir) + _corr2(fused - ir, vi))


def Qabf_function(ir, vi, fused):
    """Edge information transfer metric."""
    return float(get_Qabf(ir, vi, fused))


def _lpips_tensor(path):
    """Load an image with the official LPIPS helper."""
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requires the 'lpips' package. Install it with: pip install lpips") from exc

    return lpips.im2tensor(lpips.load_image(str(path)))


@lru_cache(maxsize=8)
def _lpips_model(net="alex", version="0.1", device="cpu"):
    """Cache the LPIPS model."""
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requires the 'lpips' package. Install it with: pip install lpips") from exc

    model = lpips.LPIPS(net=net, version=version).to(device)
    model.eval()
    return model


def LPIPS_pair_function(ref_path, fused_path, device="cpu", net="alex", version="0.1"):
    """LPIPS distance between a reference image and a fused image."""
    import torch

    model = _lpips_model(net=net, version=version, device=device)
    ref = _lpips_tensor(ref_path).to(device)
    fused = _lpips_tensor(fused_path).to(device)

    if ref.shape[-2:] != fused.shape[-2:]:
        raise ValueError("LPIPS requires paired images to have the same spatial size.")

    with torch.no_grad():
        return float(model.forward(ref, fused).item())


def LPIPS_function(ir_path, vi_path, fused_path, device="cpu", net="alex", version="0.1", ref="vi"):
    """Compute LPIPS using IR, VI, or the mean of both source-image references."""
    if ref == "ir":
        return LPIPS_pair_function(ir_path, fused_path, device=device, net=net, version=version)
    if ref == "vi":
        return LPIPS_pair_function(vi_path, fused_path, device=device, net=net, version=version)
    if ref == "mean":
        ir_score = LPIPS_pair_function(ir_path, fused_path, device=device, net=net, version=version)
        vi_score = LPIPS_pair_function(vi_path, fused_path, device=device, net=net, version=version)
        return float((ir_score + vi_score) / 2.0)
    raise ValueError("ref must be one of: ir, vi, mean")


def evaluate_image(ir_path, vi_path, fused_path, lpips_device="cpu", lpips_net="alex", lpips_version="0.1", lpips_ref="vi"):
    """Compute the eight metrics used in the paper for one image pair."""
    ir = load_gray(ir_path)
    vi = load_gray(vi_path)
    fused = load_gray(fused_path)

    if ir.shape != fused.shape or vi.shape != fused.shape:
        raise ValueError("Source and fused images must have the same grayscale size.")

    return {
        "EN": EN_function(fused.astype(np.int32)),
        "SF": SF_function(fused),
        "AG": AG_function(fused),
        "SD": SD_function(fused),
        "SCD": SCD_function(ir, vi, fused),
        "VIF": VIF_function(ir, vi, fused),
        "Qabf": Qabf_function(ir, vi, fused),
        "LPIPS": LPIPS_function(ir_path, vi_path, fused_path, device=lpips_device, net=lpips_net, version=lpips_version, ref=lpips_ref),
    }
