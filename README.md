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

| Model | Type | AUROC | Labels needed? |
|-------|------|-------|----------------|
| ResNet-18 | Supervised CNN | 0.8947 | Yes |
| EfficientNet-B0 | Supervised CNN | 0.9737 | Yes |
| ViT-Base | Supervised Transformer | 1.0000 | Yes |
| Swin-Tiny | Supervised Transformer | 0.8000 | Yes |
| DINOv2 + kNN | Unsupervised | 0.9413 | No (normal images only) |
| CLIP zero-shot | Zero-shot VLM | 0.6628 | No (nothing) |

## Key Findings

- EfficientNet-B0 achieves the best supervised balance — 0.97 AUROC and 95% defect recall, most suitable where missing a defect is catastrophic
- ViT-Base achieves perfect AUROC (1.0) on the test set but result should be validated on a larger test set (only 23 test images)
- DINOv2 + kNN achieves 0.94 AUROC with zero defect labels — beats supervised ResNet-18, demonstrating unsupervised anomaly detection is viable for label-scarce manufacturing environments
- CLIP zero-shot achieves 0.66 AUROC with no training and no examples, but performance is highly sensitive to prompt wording ("good"/"defective" outperformed longer descriptive prompts)
- Swin-Tiny collapses to predicting everything as defective — local window attention insufficient to learn class separation with only 92 training images
- All supervised models show overfitting after epoch 10 due to small dataset; transfer learning mitigates but does not eliminate this

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
├── phase3_dinov2.py           # DINOv2 + kNN unsupervised anomaly detection
├── phase4_clip.py             # CLIP zero-shot defect detection
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
pip install torch torchvision timm scikit-learn matplotlib seaborn pillow opencv-python transformers
```

Download MVTec AD from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place under `data/`.

```bash
python phase1_resnet.py
python phase1_efficientnet.py
python phase2_vit.py
python phase2_swin.py
python phase3_dinov2.py
python phase4_clip.py
```
