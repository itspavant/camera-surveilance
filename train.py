import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import cv2
import json
from datetime import datetime
from pathlib import Path
import gc
from tensorflow.keras.applications.efficientnet import preprocess_input

# Metrics and evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

os.makedirs(
    "violence_detection_results",
    exist_ok=True
)

os.makedirs(
    "violence_detection_results/plots",
    exist_ok=True
)

os.makedirs(
    "violence_detection_results/models",
    exist_ok=True
)

os.makedirs(
    "violence_detection_results/reports",
    exist_ok=True
)

os.makedirs(f"violence_detection_results/metrics",
            exist_ok=True
)

# ================= CONFIG =================
DATASET_PATH = "dataset_frames"
SEQ_LEN = 16
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
STRIDE = 8
RESULTS_DIR = "violence_detection_results"

# Create results directory
Path(RESULTS_DIR).mkdir(exist_ok=True)
Path(f"{RESULTS_DIR}/models").mkdir(exist_ok=True)
Path(f"{RESULTS_DIR}/plots").mkdir(exist_ok=True)
Path(f"{RESULTS_DIR}/metrics").mkdir(exist_ok=True)

# ================= GPU =================
gpus = tf.config.list_physical_devices('GPU')
print("GPU:", gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ================= GENERATOR =================
class FrameGenerator(Sequence):
    def __init__(self, video_dirs, labels, batch_size, img_size, training=True, model_type="generic"):
        super().__init__()
        self.video_dirs = video_dirs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.training = training
        self.model_type = model_type

        # 🔥 Preload frame lists (IMPORTANT)
        self.video_frames = {}
        for path in self.video_dirs:
            self.video_frames[path] = sorted(
                os.listdir(path),
                key=lambda x: int(x.split('.')[0])
            )

        self.samples = self.build_samples()

    def build_samples(self):
        samples = []
        for path, label in zip(self.video_dirs, self.labels):
            frames = self.video_frames[path]

            if len(frames) < SEQ_LEN:
                continue

            for i in range(0, len(frames) - SEQ_LEN, STRIDE):
                samples.append((path, i, label))

        return samples

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch = self.samples[idx*self.batch_size:(idx+1)*self.batch_size]

        X, y = [], []

        for path, start, label in batch:
            frames = self.video_frames[path][start:start+SEQ_LEN]
            clip = []

            for f in frames:
                img_path = os.path.join(path, f)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
                if self.model_type == "efficientnet":
                    img = preprocess_input(img.astype(np.float32))
                else:
                    img = img / 255.0

                # 🔥 AUGMENT ONLY TRAIN
                if self.training:
                    if np.random.rand() > 0.5:
                        img = np.fliplr(img)

                clip.append(img)

            if len(clip) == SEQ_LEN:
                X.append(clip)
                y.append(label)

        if len(X) == 0:
            raise ValueError("Empty batch encountered — check dataset")

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ================= LOAD PATHS =================
def load_dirs(split):
    paths, labels = [], []
    for label, cls in enumerate(["normal", "violent"]):
        folder = os.path.join(DATASET_PATH, split, cls)
        if not os.path.exists(folder):
            print(f"Warning: {folder} does not exist")
            continue
        for vid in os.listdir(folder):
            vid_path = os.path.join(folder, vid)
            if os.path.isdir(vid_path):
                paths.append(vid_path)
                labels.append(label)
    return paths, labels

print("Loading dataset...")
train_dirs, y_train = load_dirs("train")
val_dirs, y_val = load_dirs("val")

print(f"Training samples: {len(train_dirs)}")
print(f"Validation samples: {len(val_dirs)}")

# ================= MODELS =================
def build_model(base, bilstm=False):
    if base == "mobilenet":
        cnn = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

        for layer in cnn.layers[:-20]:
            layer.trainable = False

    elif base == "efficientnet":
        cnn = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

        # IMPORTANT CHANGE
        for layer in cnn.layers[:-50]:
            layer.trainable = False

    elif base == "resnet":
        cnn = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

        for layer in cnn.layers[:-30]:
            layer.trainable = False

    temporal = Bidirectional(LSTM(128)) if bilstm else LSTM(128)

    return Sequential([
        TimeDistributed(cnn, input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(GlobalAveragePooling2D()),
        temporal,
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

def build_3dcnn():
    return Sequential([
        Conv3D(
            32,
            (3,3,3),
            activation='relu',
            padding='same',
            input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)
        ),
        BatchNormalization(),
        MaxPooling3D((1,2,2)),

        Conv3D(
            64,
            (3,3,3),
            activation='relu',
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling3D((1,2,2)),

        Conv3D(
            128,
            (3,3,3),
            activation='relu',
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling3D((2,2,2)),

        GlobalAveragePooling3D(),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

# ================= EVALUATION FUNCTIONS =================
def evaluate_model(model, generator, model_name):
    """Comprehensive evaluation with all metrics"""
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    print(f"Evaluating {model_name}...")
    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        predictions = model.predict(X_batch, verbose=0)
        
        y_true.extend(y_batch.astype(int))
        y_pred_proba.extend(predictions.flatten())
        y_pred.extend((predictions > 0.5).astype(int).flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true, y_pred_proba)),
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])
    
    # Specificity and Sensitivity
    if (cm[0, 0] + cm[0, 1]) > 0:
        metrics['specificity'] = float(cm[0, 0] / (cm[0, 0] + cm[0, 1]))
    else:
        metrics['specificity'] = 0.0
        
    if (cm[1, 0] + cm[1, 1]) > 0:
        metrics['sensitivity'] = float(cm[1, 1] / (cm[1, 0] + cm[1, 1]))
    else:
        metrics['sensitivity'] = 0.0
    
    return metrics, y_true, y_pred, y_pred_proba, cm

def save_plots(model_name, y_true, y_pred, y_pred_proba, cm):
    """Save visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
    axes[0, 0].set_title(f'{model_name} - Confusion Matrix')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xlabel('Predicted')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title(f'{model_name} - ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid()
    
    # Prediction Distribution
    axes[1, 0].hist([y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]], 
                    bins=30, label=['Normal', 'Violent'], alpha=0.7)
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{model_name} - Prediction Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid()
    
    # Metrics Bar Chart
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    metrics_values = [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
        roc_auc_score(y_true, y_pred_proba)
    ]
    axes[1, 1].bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].set_title(f'{model_name} - Performance Metrics')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(metrics_values):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    axes[1, 1].grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plots/{model_name}_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots for {model_name}")

# ================= CONFIG =================
configs = [
    ("3D_CNN", "3d", False),
    ("EfficientNet_LSTM", "efficientnet", False),
    ("EfficientNet_BiLSTM", "efficientnet", True),
    ("MobileNet_LSTM", "mobilenet", False),
    ("MobileNet_BiLSTM", "mobilenet", True),
    ("ResNet_LSTM", "resnet", False),
    ("ResNet_BiLSTM", "resnet", True),
]

results = {}
all_results = []

# ================= TRAIN =================
for name, base, bilstm in configs:
    
    print("\n" + "="*60)
    print(f"🚀 Processing {name}")
    print("="*60)

    metrics_file = f"{RESULTS_DIR}/metrics/{name}_results.json"

    if os.path.exists(metrics_file):
        print(f"✅ Existing results found for {name}")

        with open(metrics_file, "r") as f:
            result = json.load(f)

        all_results.append(result)
        results[name] = result

        print("Skipping training...")
        print(f"Accuracy : {result['accuracy']:.4f}")
        print(f"F1 Score : {result['f1']:.4f}")
        print(f"AUC      : {result['auc_roc']:.4f}")

        continue
    
    print(f"\n{'='*60}")
    print(f"🚀 Training {name}")
    print(f"{'='*60}")
    
    try:
        # Create generators
        train_gen = FrameGenerator(
            train_dirs,
            y_train,
            BATCH_SIZE,
            IMG_SIZE,
            training=True,
            model_type=base
        )

        val_gen = FrameGenerator(
            val_dirs,
            y_val,
            BATCH_SIZE,
            IMG_SIZE,
            training=False,
            model_type=base
        )
        
        print("Train windows:", len(train_gen.samples))
        print("Val windows:", len(val_gen.samples))

        # Build model
        model = build_3dcnn() if base == "3d" else build_model(base, bilstm)
        
        print(
            "Trainable params:",
            np.sum(
                [np.prod(v.shape)
                 for v in model.trainable_weights]
            )
        )

        # Compile with optimized settings
        if base == "efficientnet":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=5e-5
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-4
            )
        
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        start = time.time()
        
        print(f"Train windows: {len(train_gen.samples)}")
        print(f"Val windows: {len(val_gen.samples)}")
        print(f"Trainable params: {model.count_params()}")
        print("STARTING FIT")

        # Training with callbacks
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=[
                EarlyStopping(patience=2, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-7),
                ModelCheckpoint(
                    f"{RESULTS_DIR}/models/{name}_best.keras",
                    save_best_only=True,
                    monitor='val_loss'
                )
            ],
            verbose=1
        )
        
        print("FIT FINISHED")

        duration = time.time() - start

        # Evaluate on validation set
        print(f"\n📊 Evaluating {name} on Validation Set...")
        val_metrics, y_true_val, y_pred_val, y_pred_proba_val, cm_val = evaluate_model(model, val_gen, name)
        
        # Save plots
        save_plots(name, y_true_val, y_pred_val, y_pred_proba_val, cm_val)

        # Compile results
        model_result = {
            "model_name": name,
            "base_model": base,
            "bilstm": bilstm,
            "training_time_sec": float(duration),
            "epochs_trained": len(history.history['loss']),
            "final_train_loss": float(history.history['loss'][-1]),
            "final_train_accuracy": float(history.history['accuracy'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        }
        
        # Add validation metrics
        model_result.update(val_metrics)
        results[name] = model_result
        all_results.append(model_result)
        
        # Save individual model results
        with open(f"{RESULTS_DIR}/metrics/{name}_results.json", 'w') as f:
            json.dump(model_result, f, indent=4)
        
        print(f"✅ {name} - Training Complete!")
        print(f"   Accuracy: {model_result['accuracy']:.4f}")
        print(f"   F1 Score: {model_result['f1']:.4f}")
        print(f"   AUC-ROC: {model_result['auc_roc']:.4f}")
        print(f"   Sensitivity: {model_result['sensitivity']:.4f}")
        print(f"   Specificity: {model_result['specificity']:.4f}")
        
        # Save model summary
        with open(f"{RESULTS_DIR}/metrics/{name}_summary.txt", 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Clean up to free memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error training {name}: {str(e)}")
        results[name] = {"error": str(e)}
        continue

# ================= SAVE COMPREHENSIVE RESULTS =================
print(f"\n{'='*60}")
print(f"📊 GENERATING COMPREHENSIVE REPORT")
print(f"{'='*60}\n")

# Create DataFrame for comparison
df_results = pd.DataFrame(all_results)

# Save to CSV
csv_path = f"{RESULTS_DIR}/final_comparison.csv"
df_results.to_csv(csv_path, index=False)
print(f"✅ Results saved to: {csv_path}")

# Save detailed JSON
json_path = f"{RESULTS_DIR}/detailed_results.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"✅ Detailed results saved to: {json_path}")

# Print summary table
print("\n" + "="*100)
print("FINAL COMPARISON TABLE")
print("="*100)

# Select key metrics for display
display_cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'sensitivity', 'specificity', 'training_time_sec']
if all(col in df_results.columns for col in display_cols):
    df_display = df_results[display_cols].copy()
    df_display = df_display.sort_values('f1', ascending=False)
    print(df_display.to_string(index=False))
else:
    print(df_results.to_string())

# Generate comparison plots
print("\n📈 Generating comparison plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Get sorted data
df_sorted = df_results.sort_values('f1', ascending=False)

metrics_to_plot = [
    ('accuracy', 'Accuracy'),
    ('f1', 'F1 Score'),
    ('auc_roc', 'AUC-ROC'),
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('training_time_sec', 'Training Time (seconds)')
]

for idx, (metric, title) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    if metric in df_sorted.columns:
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(df_sorted))]
        ax.bar(range(len(df_sorted)), df_sorted[metric], color=colors)
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['model_name'], rotation=45, ha='right')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(title)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(df_sorted[metric]):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Comparison plot saved to: {RESULTS_DIR}/model_comparison.png")

# Generate HTML report
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Violence Detection - Model Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .best {{ background-color: #d4edda; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Violence Detection Model Comparison</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Best Performing Model</h2>
        <div class="metric-box">
            <strong>Model:</strong> {df_sorted.iloc[0]['model_name']}<br>
            <strong>F1 Score:</strong> {df_sorted.iloc[0]['f1']:.4f}<br>
            <strong>Accuracy:</strong> {df_sorted.iloc[0]['accuracy']:.4f}<br>
            <strong>AUC-ROC:</strong> {df_sorted.iloc[0]['auc_roc']:.4f}
        </div>
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Model Name</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>AUC-ROC</th>
                <th>Sensitivity</th>
                <th>Specificity</th>
                <th>Training Time (s)</th>
            </tr>
"""

for _, row in df_sorted.iterrows():
    html_content += f"""
            <tr>
                <td>{row['model_name']}</td>
                <td>{row['accuracy']:.4f}</td>
                <td>{row['precision']:.4f}</td>
                <td>{row['recall']:.4f}</td>
                <td>{row['f1']:.4f}</td>
                <td>{row['auc_roc']:.4f}</td>
                <td>{row['sensitivity']:.4f}</td>
                <td>{row['specificity']:.4f}</td>
                <td>{row['training_time_sec']:.2f}</td>
            </tr>
"""

html_content += """
        </table>
        
        <h2>Visualizations</h2>
        <h3>Model Comparison</h3>
        <img src="model_comparison.png" alt="Model Comparison">
"""

# Add individual model plots
for _, row in df_sorted.iterrows():
    model_name = row['model_name']
    html_content += f"""
        <h3>{model_name}</h3>
        <img src="plots/{model_name}_evaluation.png" alt="{model_name} Evaluation">
"""

html_content += """
    </div>
</body>
</html>
"""

html_path = f"{RESULTS_DIR}/report.html"
with open(html_path, 'w') as f:
    f.write(html_content)
print(f"✅ HTML report saved to: {html_path}")

# Print final summary
print("\n" + "="*100)
print("🎉 TRAINING COMPLETE!")
print("="*100)
print(f"📁 All results saved in: {RESULTS_DIR}/")
print(f"   ├── models/          (Trained model checkpoints)")
print(f"   ├── metrics/         (Individual model metrics in JSON)")
print(f"   ├── plots/           (Evaluation plots for each model)")
print(f"   ├── final_comparison.csv")
print(f"   ├── detailed_results.json")
print(f"   ├── model_comparison.png")
print(f"   └── report.html      (HTML report)")
print("="*100)