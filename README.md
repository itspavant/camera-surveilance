# 🛡️ AI-Based Violence Detection System

An end-to-end Deep Learning framework for automated violence detection in surveillance videos using **CNN-LSTM**, **CNN-BiLSTM**, and **3D CNN** architectures.

This project performs a comparative study of seven deep learning models on the **RWF-2000** dataset to identify the most suitable architecture for real-time CCTV surveillance systems.

---

## 📌 Overview

Traditional CCTV surveillance relies heavily on continuous human monitoring, making it prone to fatigue and delayed response during critical incidents.

This project automates violence detection by analyzing short video sequences and classifying them into:

- 🔴 Violence
- 🟢 Non-Violence

Instead of relying only on spatial information from individual frames, the proposed system captures both:

- **Spatial Features** using Convolutional Neural Networks (CNNs)
- **Temporal Features** using LSTM/BiLSTM networks

A **3D CNN** model is also implemented as a baseline for comparison.

---

## 🚀 Features

- Comparative study of **7 Deep Learning architectures**
- Transfer Learning using ImageNet pretrained weights
- Sliding Window Sequence Generation
- Automatic Frame Extraction
- Custom TensorFlow/Keras Data Generator
- GPU-accelerated Training
- Early Stopping & Model Checkpointing
- Automatic Evaluation Report Generation
- Confusion Matrix & ROC Curve Visualization
- HTML Report Generation
- JSON & CSV Metrics Export

---

## 📂 Dataset

**Dataset:** RWF-2000 (Real-World Fight Dataset)

The dataset contains:

- 2000 surveillance videos
- Binary Classification
  - Violence
  - Non-Violence

Each video is converted into image frames before training.

---

## 🏗️ System Architecture

```text
Input Video
     │
     ▼
Frame Extraction
     │
     ▼
Sliding Window Generation
(16 Frames, Stride = 8)
     │
     ▼
Preprocessing
• Resize (224×224)
• RGB Conversion
• Normalization
• Data Augmentation
     │
     ▼
CNN Backbone
(MobileNetV2 / EfficientNetB0 / ResNet50)
     │
     ▼
LSTM / BiLSTM
Temporal Feature Learning
     │
     ▼
Dense + Sigmoid
Binary Classification
     │
     ▼
Violence / Non-Violence
```

---

## 🤖 Models Compared

| Model | Spatial Feature Extractor | Temporal Feature Extractor |
|--------|--------------------------|----------------------------|
| 3D CNN | 3D Convolution | Built-in |
| MobileNetV2 + LSTM | MobileNetV2 | LSTM |
| MobileNetV2 + BiLSTM | MobileNetV2 | BiLSTM |
| EfficientNetB0 + LSTM | EfficientNetB0 | LSTM |
| EfficientNetB0 + BiLSTM | EfficientNetB0 | BiLSTM |
| ResNet50 + LSTM | ResNet50 | LSTM |
| ResNet50 + BiLSTM | ResNet50 | BiLSTM |

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy |
| Batch Size | 8 |
| Epochs | 10 (Early Stopping Enabled) |
| Sequence Length | 16 Frames |
| Sliding Window Stride | 8 |
| Transfer Learning | Yes |
| Model Checkpoint | Enabled |

---

## 📊 Performance Comparison

| Model | Accuracy | F1 Score | AUC-ROC |
|--------|---------:|---------:|---------:|
| **MobileNetV2 + LSTM** | **70.53%** | **0.727** | **0.757** |
| MobileNetV2 + BiLSTM | 69.29% | 0.676 | 0.755 |
| EfficientNetB0 + LSTM | 68.06% | 0.689 | 0.741 |
| EfficientNetB0 + BiLSTM | 63.09% | 0.669 | 0.708 |
| 3D CNN | 57.88% | 0.569 | 0.623 |
| ResNet50 + BiLSTM | 58.15% | 0.514 | 0.640 |
| ResNet50 + LSTM | 52.94% | 0.280 | 0.607 |

🏆 **Best Performing Model:** **MobileNetV2 + LSTM**

---

## 📁 Project Structure

```text
Violence-Detection/
│
├── dataset/
├── dataset_frames/
├── models/
├── violence_detection_results/
│   ├── metrics/
│   ├── models/
│   ├── plots/
│   ├── report.html
│   ├── detailed_results.json
│   └── final_comparison.csv
│   └── model_comparision.png
│
├── train.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras

### Computer Vision

- OpenCV

### Data Processing

- NumPy
- Pandas

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning Utilities

- Scikit-learn

---

## 📈 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Sensitivity
- Specificity
- Confusion Matrix

---

## 📷 Outputs Generated

The framework automatically generates:

- 📊 Model Comparison Plot
- 📈 ROC Curve
- 📉 Training & Validation Curves
- 🔲 Confusion Matrix
- 📄 HTML Evaluation Report
- 📁 JSON Metrics
- 📊 CSV Comparison Table

---

## 🔮 Future Work

- Real-time CCTV surveillance integration
- Live webcam inference
- Automated alert generation
- Edge AI deployment
- Vision Transformer (ViT)
- Temporal Attention Networks
- Model quantization for embedded devices
- Multi-class violence detection

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
