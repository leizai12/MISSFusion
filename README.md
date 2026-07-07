<div align="center">

# MISSFusion: A Multi-granularity Interaction and Semantic-Structure Rectification Network for Infrared and Visible Image Fusion

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey)](#)
[![Code](https://img.shields.io/badge/Code-PyTorch-orange.svg)](#)


</div>

## 📢 Announcement

**Code Availability:** This repository is the official implementation of **MISSFusion**. The complete source code, including pre-trained models, training configurations, and evaluation scripts, is currently being organized for better reproducibility. **The full codebase will be released immediately upon the official acceptance of the paper.** We appreciate your patience and interest—stay tuned by starring ⭐ this repository!

---

## 🛠️ Installation & Requirements

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

*> Note: For downstream object detection evaluation, please ensure that the required detection dependencies are installed. Our experiments use the [YOLO26l detector](https://docs.ultralytics.com/models/yolo26/#overview).*

---

## 📂 Dataset Preparation

We evaluate MISSFusion on the following public datasets:
- **MSRS**
- **M3FD**
- **RoadScene**
- **LLVIP**
- **HDO**
- **TNO**

Please organize the datasets as follows:

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

Please modify the dataset paths in the corresponding config files before running training or testing.

---

## 💻 Quick Start

### 1. Training
To train the model from scratch on the MSRS dataset, run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_msrs.yaml
```

### 2. Inference / Testing
To test the model using our provided pre-trained weights (available after code release):

```bash
python test.py --config configs/test_msrs.yaml --checkpoint checkpoints/missfusion_msrs.pth
```

### 3. Evaluation (Objective Metrics)
To compute quantitative metrics such as EN, SF, AG, SD, SCD, VIF, and $N_{AB/F}$:

```bash
python eval.py --dataset MSRS --result_dir results/MSRS
```

---

## 📧 Contact

If you have any questions or encounter issues, please open an issue in this repository.
