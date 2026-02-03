# GeoAlign: Foundation Model-driven Asymmetric Dual-Stream Manifold Alignment for Unsupervised Cross-View Geo-Localization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.8.0-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official inference code for the paper **"GeoAlign: Foundation Model-driven Asymmetric Dual-Stream Manifold Alignment for Unsupervised Cross-View Geo-Localization"**.

**GeoAlign** is a novel unsupervised framework that leverages frozen vision foundation models (DINOv3) to address the severe geometric distortions and lack of annotations in drone-to-satellite matching.

## 📖 Introduction

Cross-view geo-localization (CVGL) faces significant challenges due to platform-specific view discrepancies and the reliance on costly paired annotations. We propose **GeoAlign**, an asymmetric dual-stream framework designed to bridge these gaps without ground-truth supervision.

Our method incorporates four core contributions:
* **Asymmetric Geometry-Rectified Adapter (AGRA):** Utilizes a Mixture-of-Experts (MoE) exclusively on the drone branch to rectify non-linear geometric distortions towards the standard satellite feature space.
* **Optimal Transport-driven Curriculum Alignment (OTCA):** Leverages the Sinkhorn algorithm to generate robust soft pseudo-labels, guiding the model from coarse global alignment to fine-grained matching.
* **Distribution-Aware Mutual Optimization (DAMO):** A hybrid strategy integrating bidirectional soft distillation and reliability-driven complex matching to ensure robust convergence.
* **Intrinsic Structure Mining Stream (ISMS):** Employs manifold-constrained whitening to eliminate channel redundancy and uncover fine-grained discriminative cues.

## 🚀 Comparison with State-of-the-Arts

GeoAlign establishes a new state-of-the-art for unsupervised CVGL, significantly outperforming existing methods and even rivaling advanced supervised techniques.

### University-1652 Benchmark

| Method | Learning | Drone → Satellite (R@1) | Drone → Satellite (AP) |
| :--- | :---: | :---: | :---: |
| EM-CVGL (TGRS'24) | Unsup. | 70.29 | 74.93 |
| Wang et al. (AAAI'25) | Unsup. | 85.95 | 90.33 |
| **GeoAlign* (Ours)** | **Unsup.** | **88.11** | **92.87** |

### SUES-200 Benchmark (150m Altitude)

| Method | Learning | Drone → Satellite (R@1) | Drone → Satellite (AP) |
| :--- | :---: | :---: | :---: |
| EM-CVGL (TGRS'24) | Unsup. | 55.23 | 60.80 |
| **GeoAlign* (Ours)** | **Unsup.** | **94.23** | **96.78** |

## 🛠️ Requirements

The code is tested with Python 3.8+ and PyTorch 2.8.0.

1. Clone this repository:
   ```bash
   git clone [https://github.com/JZLhope/GeoAlign.git](https://github.com/JZLhope/GeoAlign.git)
   cd GeoAlign

```

2. Install dependencies:
```bash
pip install -r requirements.txt

```


*> Note: Please ensure `easydict`, `pyyaml`, and `dinov3` related dependencies are correctly installed.*

## 📂 Data & Model Weights

### 1. Download Resources

We provide the pre-trained model weights (including the MoE module and ISMS autoencoder) and pre-extracted features via Baidu Netdisk.

* **Link:** [Baidu Netdisk](https://pan.baidu.com/s/1Er16FNa1j4xOphuujqguqg)
* **Code:** `1688`

### 2. Directory Structure

To ensure the evaluation scripts run seamlessly, please organize the downloaded files as follows:

```text
GeoAlign/
├── configs/             # Configuration files
├── data/
│   └── University-Release/  # U1652 Dataset images (if extracting features)
├── feats_test/          # Pre-extracted D2S features
├── feats_test_s2d/      # Pre-extracted S2D features
├── outputs/
│   ├── base_stream1/    # Contains '300_param.t' (MoE weights)
│   └── base_stream2/    # Contains 'isms_model_26_28.pth' (ISMS weights)
└── ...

```

## ⚡ Inference

The inference pipeline consists of **Feature Extraction** (optional if using provided features) and **Evaluation**.

### Step 1: Feature Extraction

Extract features using the frozen DINOv3 backbone (ViT-H+/16).

**1. Extract Test Features (Drone -> Satellite):**

```bash
python -m extract_and_save configs/base_dinov3_extract_D2S.yml \
  --model_name dinov3_vith16plus \
  --save_dir ./feats_test \
  --desc_layer 26 28

```

**2. Extract Test Features (Satellite -> Drone):**

```bash
python -m extract_and_save configs/base_dinov3_extract_S2D.yml \
  --model_name dinov3_vith16plus \
  --S2D \
  --save_dir ./feats_test_s2d \
  --desc_layer 26 28

```

---

### Step 2: Evaluation

Evaluate the retrieval performance. The default setting uses the full **Dual-Stream** architecture (Fusion of Stream 1 & Stream 2).

#### Drone-to-Satellite (D2S) Evaluation

**Run GeoAlign* (Full Model with Fusion):**

```bash
python -m evaluate \
  --gpu 0 \
  --isms_path ./outputs/base_stream2/isms_model_26_28.pth

```

**Run GeoAlign (Stream 1 only / No Fusion):**

```bash
python -m evaluate --no_fusion --gpu 0

```

#### Satellite-to-Drone (S2D) Evaluation

**Run GeoAlign* (Full Model with Fusion):**

```bash
python -m evaluate \
  --mode S2D \
  --gpu 0 \
  --isms_path ./outputs/base_stream2/isms_model_26_28.pth

```

## 🎓 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{Liu2026GeoAlign,
  title={GeoAlign: Foundation Model-driven Asymmetric Dual-Stream Manifold Alignment for Unsupervised Cross-View Geo-Localization},
  author={Liu, Juzheng and Qin, Hanlin and Zhang, Xupei and Pang, Zibo and Deng, Chenguang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology (Submitted)},
  year={2026}
}

```

## 🙏 Acknowledgements

This work utilizes resources from [DINOv3](https://github.com/facebookresearch/dinov3). We thank the authors for their open-source contribution.

```

```
