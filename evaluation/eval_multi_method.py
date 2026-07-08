import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from Metric import METRIC_NAMES, evaluate_image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the eight metrics reported in the MISSFusion paper.")
    parser.add_argument("--ir_dir", required=True, help="Directory of infrared source images.")
    parser.add_argument("--vi_dir", required=True, help="Directory of visible source images.")
    parser.add_argument("--fused_root", required=True, help="Directory containing fused images or method subdirectories.")
    parser.add_argument("--methods", nargs="*", default=None, help="Method names under fused_root. If omitted, subdirectories are used.")
    parser.add_argument("--output", default="metric_results.xlsx", help="Output Excel file.")
    parser.add_argument("--lpips_device", default="auto", choices=("auto", "cpu", "cuda"), help="Device for LPIPS.")
    parser.add_argument("--lpips_net", default="alex", choices=("alex", "vgg", "squeeze"), help="Backbone used by LPIPS.")
    parser.add_argument("--lpips_version", default="0.1", help="LPIPS model version.")
    parser.add_argument("--lpips_ref", default="vi", choices=("ir", "vi", "mean"), help="Reference used for LPIPS in fusion evaluation.")
    parser.add_argument("--round", type=int, default=4, help="Number of decimals in saved results.")
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


def lpips_device(name):
    if name != "auto":
        return name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def summarize(rows, decimals):
    summary_rows = []
    for method, group in rows.groupby("Method", sort=False):
        mean_row = {"Method": method, "File": "mean"}
        std_row = {"Method": method, "File": "std"}
        for metric in METRIC_NAMES:
            mean_row[metric] = group[metric].mean()
            std_row[metric] = group[metric].std(ddof=0)
        summary_rows.extend([mean_row, std_row])
    return pd.DataFrame(summary_rows).round(decimals)


def main():
    args = parse_args()
    ir_dir = Path(args.ir_dir)
    vi_dir = Path(args.vi_dir)
    fused_root = Path(args.fused_root)
    output = Path(args.output)
    device = lpips_device(args.lpips_device)

    ir_files = list_images(ir_dir)
    if not ir_files:
        raise FileNotFoundError(f"No images found in {ir_dir}")

    methods = discover_methods(fused_root, args.methods)
    records = []

    for method in methods:
        fused_dir = method_directory(fused_root, method)
        progress = tqdm(ir_files, desc=method)
        for ir_path in progress:
            vi_path = resolve_by_name(vi_dir, ir_path.name)
            fused_path = resolve_by_name(fused_dir, ir_path.name)
            if vi_path is None:
                print(f"Warning: missing visible image for {ir_path.name}. Skipped.")
                continue
            if fused_path is None:
                print(f"Warning: missing fused image for {method}/{ir_path.name}. Skipped.")
                continue

            try:
                metrics = evaluate_image(
                    ir_path,
                    vi_path,
                    fused_path,
                    lpips_device=device,
                    lpips_net=args.lpips_net,
                    lpips_version=args.lpips_version,
                    lpips_ref=args.lpips_ref,
                )
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
    print("Metrics:", ", ".join(METRIC_NAMES))
    print("Note: LPIPS is lower better; the other metrics are higher better.")


if __name__ == "__main__":
    main()



