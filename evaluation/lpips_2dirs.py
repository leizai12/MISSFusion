import argparse
from pathlib import Path

import lpips
import torch
from natsort import natsorted

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute average LPIPS between two image directories.")
    parser.add_argument("--dir0", required=True, help="Reference image directory.")
    parser.add_argument("--dir1", required=True, help="Compared image directory.")
    parser.add_argument("--net", default="alex", choices=("alex", "vgg", "squeeze"), help="LPIPS backbone.")
    parser.add_argument("--version", default="0.1", help="LPIPS model version.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Device used for LPIPS.")
    return parser.parse_args()


def choose_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_images(directory):
    directory = Path(directory)
    return [p for p in natsorted(directory.iterdir(), key=lambda x: x.name) if p.suffix.lower() in IMAGE_EXTENSIONS]


def find_pair(directory, filename):
    directory = Path(directory)
    direct = directory / filename
    if direct.exists():
        return direct
    stem = Path(filename).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    args = parse_args()
    device = choose_device(args.device)
    loss_fn = lpips.LPIPS(net=args.net, version=args.version).to(device)
    loss_fn.eval()

    files = list_images(args.dir0)
    total = 0.0
    count = 0

    for ref_path in files:
        cmp_path = find_pair(args.dir1, ref_path.name)
        if cmp_path is None:
            print(f"Warning: missing paired image for {ref_path.name}. Skipped.")
            continue

        try:
            img0 = lpips.im2tensor(lpips.load_image(str(ref_path))).to(device)
            img1 = lpips.im2tensor(lpips.load_image(str(cmp_path))).to(device)
            with torch.no_grad():
                score = float(loss_fn.forward(img0, img1).item())
        except Exception as exc:
            print(f"Warning: failed to process {ref_path.name}: {exc}")
            continue

        total += score
        count += 1
        print(f"{ref_path.name}: {score:.4f}")

    if count == 0:
        raise RuntimeError("No valid image pairs were processed.")

    print(f"Processed images: {count}")
    print(f"Average LPIPS distance: {total / count:.4f}")


if __name__ == "__main__":
    main()
