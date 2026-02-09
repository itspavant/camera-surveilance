# AI-Based Violence Detection System

This repository contains the **backend implementation** of an AI-powered **video-based violence detection system**, developed as part of a **BTech Major Project**.

The system uses a **CNN + LSTM (spatio-temporal deep learning model)** to analyze video streams and detect violent or suspicious activity in real time.


## 📌 Project Overview

The backend performs **video-level inference** using a sliding window of frames.  
Unlike frame-based approaches, this system captures **temporal motion patterns**, making it more robust for real-world surveillance scenarios.

### Key Capabilities
- Video-based violence detection (not frame-wise)
- Real-time webcam monitoring
- Offline video file analysis
- Alert system with cooldown mechanism
- CPU-optimized inference for real-time performance


## 🧠 Model Architecture

- **Base CNN:** MobileNetV2 (Transfer Learning)
- **Temporal Modeling:** LSTM
- **Input:** Sequence of 16 frames (224 × 224 RGB)
- **Output:** Probability of violence (binary classification)

```

Video Frames → CNN (per frame) → Temporal Aggregation (LSTM) → Classification

```


## 📂 Repository Structure

```

.
├── backend/
│   └── live_monitor.py        # Real-time video inference script
│
├── models/
│   └── violence_cnn_lstm.keras  # Trained CNN + LSTM model weights
├── videos/
│   └── normal/                  # Normal videos
│   └── violent/                 # Violent videos
│
├── requirements.txt           # Python dependencies
└── README.md

````
Link to Kaggle Notebook: [Kaggle Notebook](https://www.kaggle.com/code/tpavanteja/notebook1f9f1f599d)

## ⚙️ Installation & Setup

> Recommended Python version: 3.10 or 3.11

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```


## ▶️ Running the System

### Option 1: Webcam-Based Live Monitoring

Edit in `backend/live_monitor.py`:

```python
IS_WEBCAM = True
```

Then run:

```bash
python backend/live_monitor.py
```


### Option 2: Video File Analysis

Edit in `backend/live_monitor.py`:

```python
IS_WEBCAM = False
cap = cv2.VideoCapture("path/to/video.avi")
```

Then run:

```bash
python backend/live_monitor.py
```

## 🚨 Alert Mechanism

* Predictions are made using **video sequences**, not single frames
* Alerts are triggered only if the predicted probability crosses a threshold
* A cooldown timer prevents repeated alert spam


## Performance Optimization

To ensure smooth real-time performance on CPU:

* Inference is performed **periodically**, not on every frame
* Webcam and video files are handled differently
* Temporal predictions are reused between inference steps

This design closely mirrors **real-world surveillance systems**.


## 📊 Evaluation Summary

* Dataset used: **RWF-2000 (Real World Fights Dataset)**
* Evaluation type: **Video-level classification**
* Accuracy: ~77%
* Balanced precision and recall for violent activity detection


## ⚠️ Disclaimer

> This system is intended for **academic and research purposes only**.
> It provides **preliminary alerts** and should not be used as a replacement for professional security systems.
