import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from AG import compute_ag
from EN import compute_en, load_gray
from LPIPS import compute_lpips
from Qabf import compute_qabf
from SCD import compute_scd
from SD import compute_sd
from SF import compute_sf
from VIF import compute_vif

METRIC_NAMES = ("EN", "SF", "AG", "SD", "SCD", "VIF", "Qabf", "LPIPS")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MISSFusion paper metrics.")
    parser.add_argument("--ir_dir", required=True, help="Infrared source image directory.")
    parser.add_argument("--vi_dir", required=True, help="Visible source image directory.")
    parser.add_argument("--fused_root", required=True, help="Fused image root or method folders.")
    parser.add_argument("--methods", nargs="*", default=None, help="Method names under fused_root.")
    parser.add_argument("--output", default="metric_results.xlsx", help="Output Excel file.")
    parser.add_argument("--lpips_ref", default="vi", choices=("ir", "vi", "mean"), help="LPIPS reference source.")
    parser.add_argument("--lpips_device", default="auto", choices=("auto", "cpu", "cuda"), help="LPIPS device.")
    parser.add_argument("--lpips_net", default="alex", choices=("alex", "vgg", "squeeze"), help="LPIPS backbone.")
    parser.add_argument("--lpips_version", default="0.1", help="LPIPS model version.")
    parser.add_argument("--round", type=int, default=4, help="Saved decimal places.")
    return parser.parse_args()


def list_images(directory):
    directory = Path(directory)
    return [p for p in natsorted(directory.iterdir(), key=lambda x: x.name) if p.suffix.lower() in IMAGE_EXTENSIONS]


def resolve_by_name(directory, reference_name):
    directory = Path(directory)
    direct = directory / reference_name
    if direct.exists():
        return direct
    stem = Path(reference_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_methods(fused_root, methods):
    fused_root = Path(fused_root)
    if methods:
        return methods
    subdirs = [p.name for p in natsorted(fused_root.iterdir(), key=lambda x: x.name) if p.is_dir()]
    return subdirs if subdirs else [fused_root.name]


def method_directory(fused_root, method):
    fused_root = Path(fused_root)
    subdir = fused_root / method
    return subdir if subdir.exists() else fused_root


def choose_device(name):
    if name != "auto":
        return name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def compute_lpips_for_sources(ir_path, vi_path, fused_path, args, device):
    if args.lpips_ref == "ir":
        return compute_lpips(ir_path, fused_path, device=device, net=args.lpips_net, version=args.lpips_version)
    if args.lpips_ref == "vi":
        return compute_lpips(vi_path, fused_path, device=device, net=args.lpips_net, version=args.lpips_version)
    ir_score = compute_lpips(ir_path, fused_path, device=device, net=args.lpips_net, version=args.lpips_version)
    vi_score = compute_lpips(vi_path, fused_path, device=device, net=args.lpips_net, version=args.lpips_version)
    return float((ir_score + vi_score) / 2.0)


def evaluate_one(ir_path, vi_path, fused_path, args, device):
    ir = load_gray(ir_path)
    vi = load_gray(vi_path)
    fused = load_gray(fused_path)
    if ir.shape != fused.shape or vi.shape != fused.shape:
        raise ValueError("Source and fused images must have the same grayscale size.")

    return {
        "EN": compute_en(fused.astype(np.int32)),
        "SF": compute_sf(fused),
        "AG": compute_ag(fused),
        "SD": compute_sd(fused),
        "SCD": compute_scd(ir, vi, fused),
        "VIF": compute_vif(ir, vi, fused),
        "Qabf": compute_qabf(ir, vi, fused),
        "LPIPS": compute_lpips_for_sources(ir_path, vi_path, fused_path, args, device),
    }


def summarize(per_image, decimals):
    rows = []
    for method, group in per_image.groupby("Method", sort=False):
        mean_row = {"Method": method, "File": "mean"}
        std_row = {"Method": method, "File": "std"}
        for metric in METRIC_NAMES:
            mean_row[metric] = group[metric].mean()
            std_row[metric] = group[metric].std(ddof=0)
        rows.extend([mean_row, std_row])
    return pd.DataFrame(rows).round(decimals)


def main():
    args = parse_args()
    ir_dir = Path(args.ir_dir)
    vi_dir = Path(args.vi_dir)
    fused_root = Path(args.fused_root)
    output = Path(args.output)
    device = choose_device(args.lpips_device)

    ir_files = list_images(ir_dir)
    if not ir_files:
        raise FileNotFoundError(f"No images found in {ir_dir}")

    records = []
    for method in discover_methods(fused_root, args.methods):
        fused_dir = method_directory(fused_root, method)
        for ir_path in tqdm(ir_files, desc=method):
            vi_path = resolve_by_name(vi_dir, ir_path.name)
            fused_path = resolve_by_name(fused_dir, ir_path.name)
            if vi_path is None or fused_path is None:
                print(f"Warning: missing pair for {method}/{ir_path.name}. Skipped.")
                continue
            try:
                metrics = evaluate_one(ir_path, vi_path, fused_path, args, device)
            except Exception as exc:
                print(f"Warning: failed to evaluate {method}/{ir_path.name}: {exc}")
                continue
            records.append({"Method": method, "File": ir_path.name, **metrics})

    if not records:
        raise RuntimeError("No valid image pairs were evaluated.")

    per_image = pd.DataFrame(records).round(args.round)
    summary = summarize(per_image, args.round)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        per_image.to_excel(writer, sheet_name="per_image", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
