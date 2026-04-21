# Visual Defect Detection

Computer vision pipeline for industrial defect detection on the [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset. Benchmarks CNN and Transformer architectures spanning supervised, unsupervised, and zero-shot paradigms.

## Problem

Manufacturing defect detection requires identifying anomalies in high-resolution product images. This project benchmarks vision architectures on MVTec AD with a focus on practical tradeoffs: label requirements, AUROC, and defect recall.

## Dataset

**MVTec AD** — `metal_nut` category (115 images: 22 normal, 93 defective across 4 defect types: bent, color, flip, scratch).

Stratified 80/20 train/test split to preserve class distribution across both sets.

## Image Processing Pipeline

Each image is preprocessed using OpenCV before model inference:
- **Gaussian blur** — noise reduction before feature extraction
- **Canny edge detection** — highlights cracks, scratches, and structural boundaries
- **Adaptive thresholding** — isolates regions of interest
- **Weighted combination** — blends original RGB + edge + threshold signals

## Results

| Model | Type | Accuracy | AUROC | Defect Recall | Normal Recall |
|-------|------|----------|-------|---------------|---------------|
| ResNet-18 | Supervised CNN | 83% | 0.8947 | 0.95 | 0.25 |
| EfficientNet-B0 | Supervised CNN | 87% | 0.9737 | 0.95 | 0.50 |
| ViT-Base | Supervised Transformer | 91% | 1.0000 | 0.89 | 1.00 |
| Swin-Tiny | Supervised Transformer | 83% | 0.8000 | 1.00 | 0.00 |

## Key Findings

- EfficientNet-B0 achieves the best balance — 0.97 AUROC and 95% defect recall, making it most suitable for manufacturing where missing a defect is catastrophic
- ViT-Base achieves perfect AUROC (1.0) on the test set but lower defect recall (0.89) than EfficientNet; result should be validated on a larger test set
- Swin-Tiny collapses to predicting everything as defective — local window attention insufficient to learn class separation with only 92 training images
- All models show overfitting after epoch 10 due to small dataset (92 training images); transfer learning mitigates but does not eliminate this
- ResNet-18 and Swin-Tiny show identical accuracy (83%) for different reasons — ResNet catches more normals, Swin catches more defects

## Project Structure

```
├── explore.py                 # Dataset exploration and visualization
├── mnist.py                   # MLP baseline on MNIST
├── cnn_mnist.py               # CNN baseline on MNIST
├── preprocess.py              # OpenCV preprocessing pipeline
├── phase1_resnet.py           # ResNet-18 transfer learning
├── phase1_efficientnet.py     # EfficientNet-B0 transfer learning
├── phase2_vit.py              # ViT-Base fine-tuning
├── phase2_swin.py             # Swin-Tiny fine-tuning
├── resnet_curves.png          # ResNet training/test curves
├── resnet_confusion.png       # ResNet confusion matrix
├── efficientnet_curves.png    # EfficientNet training/test curves
├── efficientnet_confusion.png # EfficientNet confusion matrix
├── vit_curves.png             # ViT training/test curves
├── vit_confusion.png          # ViT confusion matrix
├── swin_curves.png            # Swin training/test curves
└── swin_confusion.png         # Swin confusion matrix
```

## Setup

```bash
git clone https://github.com/kmaurya9/visual-defect-detection-
cd visual-defect-detection-
python -m venv venv
source venv/bin/activate
pip install torch torchvision timm scikit-learn matplotlib seaborn pillow opencv-python
```

Download MVTec AD from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place under `data/`.

```bash
python phase1_resnet.py
python phase1_efficientnet.py
python phase2_vit.py
python phase2_swin.py
```
