# GeoAlign: Foundation Model-driven Asymmetric Dual-Stream Manifold Alignment for Unsupervised Cross-View Geo-Localization

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg)](https://pytorch.org/)
[![DINOv3](https://img.shields.io/badge/Backbone-DINOv3-blueviolet)](https://github.com/facebookresearch/dinov3)

This repository contains the official inference code for the paper **"GeoAlign: Foundation Model-driven Asymmetric Dual-Stream Manifold Alignment for Unsupervised Cross-View Geo-Localization"**.

**GeoAlign** is a novel unsupervised framework that leverages frozen vision foundation models (DINOv3) to address the severe geometric distortions in Drone-to-Satellite matching. It introduces an **Asymmetric Geometry-Rectified Adapter (AGRA)** to rectify drone-view manifolds and an **Intrinsic Structure Mining Stream (ISMS)** to distill discriminative representations via manifold-constrained whitening.

> **Note:** We currently release the **inference code and pre-trained weights**. Full training scripts will be available soon.

---

## 🚀 News
- **[2026-02]**: Inference code and pre-trained models are released. Our method achieves **88.11% R@1** on University-1652 (Drone->Satellite) without any supervision.

## 🏗️ Framework Overview

Our framework consists of two parallel streams:
1.  **Geometry-Rectified Stream (Stream 1):** Utilizes an **Asymmetric Geometry-Rectified Adapter (AGRA)** with a Mixture-of-Experts (MoE) module on the drone branch to rectify non-linear geometric distortions.
2.  **Intrinsic Structure Mining Stream (Stream 2 - ISMS):** Mines compact intrinsic structures via statistical independence and orthogonality constraints to eliminate channel redundancy.

Optimization is guided by **Optimal Transport-driven Curriculum Alignment (OTCA)** and **Distribution-Aware Mutual Optimization (DAMO)**.

## 🛠️ Requirements

- Linux
- Python 3.8+
- PyTorch 2.8.0+
- CUDA 12.0+



1. Clone the repository:
```bash
git clone https://github.com/JZLhope/GeoAlign.git
cd GeoAlign
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


*Dependencies include: `easydict`, `numpy`, `opencv-python`, `transformers`, `torch`, `torchvision`, etc.*

## 📂 Data & Model Zoo

### 1. Download Pre-trained Weights

We provide the pre-trained weights for the MoE module (Stream 1) and the ISMS module (Stream 2), along with pre-extracted features for quick evaluation.

* **Download Link (Baidu Netdisk):** [Click Here](https://pan.baidu.com/s/1Er16FNa1j4xOphuujqguqg)
* **Extraction Code:** `1688`

### 2. Prepare Directory Structure

To use the provided scripts directly, we recommend organizing your directory as follows:

```text
GeoAlign/
├── configs/             # Configuration files (included in repo)
├── data/
│   └── University-Release/  # U1652 Dataset images (if extracting features manually)
├── feats_test/          # Pre-extracted D2S features (from Netdisk)
├── feats_test_s2d/      # Pre-extracted S2D features (from Netdisk)
├── outputs/
│   ├── base_stream1/    # Place '300_param.t' here
│   └── base_stream2/    # Place 'isms_model_26_28.pth' here
├── ...

```

## ⚡ Inference Pipeline

The inference process involves two steps: **Feature Extraction** (optional if using pre-extracted features) and **Evaluation**.

### Step 1: Feature Extraction (DINOv3)

Extract features from the frozen DINOv3 backbone.

**Extract Test Features (Drone -> Satellite):**

```bash
python -m extract_and_save configs/base_dinov3_extract_D2S.yml \
  --model_name dinov3_vith16plus \
  --save_dir ./feats_test \
  --desc_layer 26 28

```

**Extract Test Features (Satellite -> Drone):**

```bash
python -m extract_and_save configs/base_dinov3_extract_S2D.yml \
  --model_name dinov3_vith16plus \
  --S2D \
  --save_dir ./feats_test_s2d \
  --desc_layer 26 28

```

---

### Step 2: Evaluation

Evaluate the retrieval performance. You can choose to evaluate the single stream (Geometry-Rectified only) or the full fused model (GeoAlign*).

#### 1. Drone-to-Satellite (D2S)

**Option A: Full Model (Fusion of Stream 1 + ISMS)** - *Recommended*

```bash
python -m evaluate \
  --gpu 0 \
  --isms_path ./outputs/base_stream2/isms_model_26_28.pth

```

**Option B: Single Stream (Stream 1 only)**

```bash
python -m evaluate --no_fusion --gpu 0

```

#### 2. Satellite-to-Drone (S2D)

**Option A: Full Model**

```bash
python -m evaluate \
  --mode S2D \
  --gpu 0 \
  --isms_path ./outputs/base_stream2/isms_model_26_28.pth

```

**Option B: Single Stream**

```bash
python -m evaluate --mode S2D --no_fusion --gpu 0

```

*Note: Ensure the paths in `evaluate.py` (specifically `MOE_WEIGHT_PATH`) point to your local `./outputs/base_stream1/300_param.t` if you moved them.*

## 📊 Main Results

Comparison with state-of-the-art methods on **University-1652**:

| Method | Type | Drone → Satellite (R@1) | Satellite → Drone (R@1) |
| --- | --- | --- | --- |
| EM-CVGL (TGRS'24) | Unsupervised | 70.29 | 79.03 |
| Wang et al. (AAAI'25) | Unsupervised | 85.95 | 94.01 |
| **GeoAlign (Ours)** | **Unsupervised** | **88.11** | **95.44** |

*Our method significantly outperforms existing unsupervised baselines and rivals advanced supervised methods.*

## 🙏 Acknowledgements

This work is built upon [DINOv3](https://github.com/facebookresearch/dinov3). We thank the authors for their open-source contribution.

