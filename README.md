---
title: NeuroScan AI - Brain Tumor Classification
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Classify brain tumors from MRI scans using deep learning
---

# 🧠 NeuroScan AI — Brain Tumor MRI Classification

A deep learning web application that classifies brain MRI scans into **4 categories**:

| Category | Description | Severity |
|----------|-------------|----------|
| 🔴 **Glioma** | Tumors from glial cells | High |
| 🟠 **Meningioma** | Tumors from meninges | Low–Moderate |
| 🟡 **Pituitary** | Pituitary gland tumors | Low–Moderate |
| ✅ **No Tumor** | Healthy brain scan | None |

## Architecture

- **Model**: EfficientNet-B0 (transfer learning from ImageNet)
- **Parameters**: ~4.4M total, fine-tuned final layers
- **Input**: 224×224 RGB MRI images
- **Training**: AdamW optimizer, cosine annealing LR, data augmentation
- **Framework**: PyTorch + Gradio

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and extract to `./dataset/`.

### 3. Train the Model
```bash
python train.py --data_dir ./dataset --epochs 20 --batch_size 32
```

### 4. Evaluate
```bash
python evaluate.py --data_dir ./dataset --model_path ./model/brain_tumor_classifier.pth
```

### 5. Launch Web App
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

## Deploy to Hugging Face Spaces (FREE)

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space → Select **Gradio** SDK
3. Upload: `app.py`, `requirements.txt`, `brain_tumor_classifier.pth`, `README.md`
4. Your app goes live with a public URL! 🎉

## Project Structure
```
NN_ABL/
├── app.py                  # Gradio web application
├── train.py                # Model training script
├── evaluate.py             # Evaluation & metrics
├── requirements.txt        # Python dependencies
├── requirements_hf.txt     # Hugging Face Spaces dependencies
├── README.md               # This file
├── dataset/                # MRI images (download from Kaggle)
│   ├── Training/
│   └── Testing/
├── model/                  # Saved model weights
│   └── brain_tumor_classifier.pth
└── evaluation/             # Evaluation outputs
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    └── evaluation_results.json
```

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not intended for clinical diagnosis. Always consult a qualified medical professional.

## License

MIT License
