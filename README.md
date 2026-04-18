# Visual Defect Detection

Computer vision pipeline for industrial defect detection on the [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset using transfer learning and OpenCV-based image preprocessing.

## Problem

Manufacturing defect detection requires identifying anomalies in high-resolution product images. This project benchmarks CNN-based transfer learning approaches on the MVTec AD dataset with a focus on practical tradeoffs: label requirements, AUROC, and defect recall.

## Dataset

**MVTec AD** — `metal_nut` category (115 images: 22 normal, 93 defective across 4 defect types: bent, color, flip, scratch).

Stratified 80/20 train/test split to preserve class distribution across both sets.

## Image Processing Pipeline

Each image is preprocessed using OpenCV before model inference:
- **Gaussian blur** — noise reduction before feature extraction
- **Canny edge detection** — highlights cracks, scratches, and structural boundaries
- **Adaptive thresholding** — isolates regions of interest
- **Weighted combination** — blends original RGB + edge + threshold signals

## Models

### ResNet-18 (Transfer Learning)
Pretrained on ImageNet. Final layer replaced with `Linear(512, 2)` for binary classification. Early layers frozen.

| Metric | Value |
|--------|-------|
| Accuracy | 83% |
| AUROC | 0.8947 |
| Defect Recall | 0.95 |
| Normal Recall | 0.25 |

### EfficientNet-B0 (Transfer Learning)
Pretrained on ImageNet. Final classifier replaced with `Linear(1280, 2)`. Early layers frozen.

| Metric | Value |
|--------|-------|
| Accuracy | 87% |
| AUROC | 0.9737 |
| Defect Recall | 0.95 |
| Normal Recall | 0.50 |

## Key Findings

- EfficientNet-B0 outperforms ResNet-18 on AUROC (0.97 vs 0.89) with less overfitting, validating compound scaling on small datasets
- Both models achieve 95% defect recall — catching 18 of 19 defects in the test set
- ResNet-18 shows overfitting after epoch 10; EfficientNet generalizes better across 20 epochs
- High false alarm rate on normal images due to class imbalance (81% defective in dataset)

## Project Structure

```
├── explore.py                 # Dataset exploration and visualization
├── mnist.py                   # MLP baseline on MNIST
├── cnn_mnist.py               # CNN baseline on MNIST
├── preprocess.py              # OpenCV preprocessing pipeline
├── phase1_resnet.py           # ResNet-18 transfer learning
├── phase1_efficientnet.py     # EfficientNet-B0 transfer learning
├── resnet_curves.png          # ResNet training/test loss and accuracy curves
├── resnet_confusion.png       # ResNet confusion matrix
├── efficientnet_curves.png    # EfficientNet training/test loss and accuracy curves
└── efficientnet_confusion.png # EfficientNet confusion matrix
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/visual-defect-detection
cd visual-defect-detection
python -m venv venv
source venv/bin/activate
pip install torch torchvision scikit-learn matplotlib seaborn pillow opencv-python
```

Download MVTec AD from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place under `data/`.

```bash
python phase1_resnet.py
python phase1_efficientnet.py
```
