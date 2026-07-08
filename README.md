<div align="center">

# MISSFusion: A Multi-Granularity Interaction and Semantic-Structure Rectification Network for Infrared and Visible Image Fusion

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey)](#)
[![Code](https://img.shields.io/badge/Code-PyTorch-orange.svg)](#)

</div>

## 📢 Current Repository Status

This repository provides the current public materials for **MISSFusion**, including evaluation scripts and test image organization for reproducibility checking.

The complete training and inference code, pretrained weights, configuration files, and detailed running instructions are being organized and will be released after the paper is officially accepted.

---

## 📁 Repository Structure

```text
MISSFusion/
├── evaluation/          # Objective metric evaluation scripts
├── test_images/         # Test image organization used by the evaluation scripts
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🛠️ Installation

The codebase has been developed and tested under the following environment:

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/leizai12/MISSFusion.git
cd MISSFusion

# Create and activate a conda environment
conda create -n missfusion python=3.9 -y
conda activate missfusion

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Organization

The test image folders are organized as follows:

```text
test_images/
├── MSRS/
│   ├── ir/
│   ├── vi/
│   └── labels/
├── M3FD/
│   ├── ir/
│   ├── vi/
│   └── labels/
├── RoadScene/
│   ├── ir/
│   └── vi/
├── LLVIP/
│   ├── ir/
│   └── vi/
├── HDO/
│   ├── ir/
│   └── vi/
└── TNO/
    ├── ir/
    └── vi/
```

Please update the image paths in the evaluation scripts if your local dataset or result directory differs from the default organization.

---

## 📊 Evaluation Scripts

The `evaluation/` directory contains scripts for objective fusion metrics, including EN, SF, AG, SD, SCD, VIF, $Q_{abf}$, and LPIPS.

Run the evaluator with source-image folders and a fused-result folder. If `--methods` is omitted, each subdirectory under `--fused_root` is treated as one method.

```bash
cd evaluation
python eval_multi_method.py \
  --ir_dir ../test_images/MSRS/ir \
  --vi_dir ../test_images/MSRS/vi \
  --fused_root ../results/MSRS \
  --output ../metrics_msrs.xlsx \
  --lpips_ref vi
```

LPIPS is computed pairwise using `lpips.LPIPS(net="alex", version="0.1")`. In the multi-metric evaluator, `--lpips_ref vi` uses the visible source image as the LPIPS reference; it can be changed to `ir` or `mean` if needed. LPIPS is lower better; the other seven metrics are higher better.

For standalone two-directory LPIPS evaluation:

```bash
cd evaluation
python lpips_2dirs.py --dir0 ../test_images/MSRS/vi --dir1 ../results/MSRS/OURS
```

---

## 🚧 Planned Release

The following materials will be released after the paper is officially accepted:

- Training code
- Inference and testing code
- Configuration files
- Pretrained weights
- Detailed instructions for reproducing the main experimental results

---

## 📧 Contact

If you have any questions or encounter issues, please open an issue in this repository.



