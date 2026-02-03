# GeoAlign: Cross-View Geo-Localization with DINOv3

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![DINOv3](https://img.shields.io/badge/Backbone-DINOv3-green)](https://github.com/facebookresearch/dinov3)

**GeoAlign** is a cross-view geo-localization framework leveraging the power of **DINOv3** vision transformers. It aims to accurately match drone-view images with satellite imagery using a multi-stream architecture involving Mixture of Experts (MoE) and Unsupervised Fusion strategies.

> **Note**: This repository currently focuses on the **inference** and evaluation protocols. Pre-trained weights and extracted features are provided to reproduce the results.

## 🚀 News
- **[2026-02]**: Code released for inference and evaluation on the U1652 dataset.

## 📂 Project Structure
The project consists of three main stages:
1.  **Feature Extraction**: Extracting features from DINOv3 layers (e.g., Layer 26, 28).
2.  **Training (Optional)**: Training the MoE AutoEncoder (Stream 1) and Fusion Model (Stream 2).
3.  **Evaluation**: Testing the retrieval performance (Drone $\leftrightarrow$ Satellite).

## 📥 Model Zoo & Features
To facilitate quick reproduction of our results, we provide the pre-trained model weights and pre-extracted features.

| Description | Link | Access Code |
| :--- | :---: | :---: |
| **Pre-trained Weights & Features** | [**Baidu Netdisk**](https://pan.baidu.com/s/1Er16FNa1j4xOphuujqguqg) | **1688** |

*Please download the files and place them in the appropriate directories (e.g., `outputs/` or `feats/`) as defined in the config files.*

## 🛠️ Requirements

* Python 3.8+
* PyTorch >= 2.0
* `torchvision`, `transformers`, `numpy`, `pillow`, `pyyaml`, `tqdm`

Install dependencies via:
```bash
pip install -r requirements.txt