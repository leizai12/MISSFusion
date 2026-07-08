"""LPIPS metric wrapper using the official lpips package."""

from functools import lru_cache

import lpips
import torch


@lru_cache(maxsize=8)
def _lpips_model(net="alex", version="0.1", device="cpu"):
    """Load and cache the LPIPS model."""
    # Cache the model to avoid reloading weights for every image pair.
    model = lpips.LPIPS(net=net, version=version).to(device)
    model.eval()
    return model


def compute_lpips(ref_path, fused_path, device="cpu", net="alex", version="0.1"):
    """Compute LPIPS distance between two images."""
    model = _lpips_model(net=net, version=version, device=device)

    # Use the official LPIPS image loader and tensor conversion.
    ref = lpips.im2tensor(lpips.load_image(str(ref_path))).to(device)
    fused = lpips.im2tensor(lpips.load_image(str(fused_path))).to(device)

    if ref.shape[-2:] != fused.shape[-2:]:
        raise ValueError("LPIPS requires paired images to have the same spatial size.")

    with torch.no_grad():
        return float(model.forward(ref, fused).item())
