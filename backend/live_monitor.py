import cv2
import numpy as np
import tensorflow as tf
import winsound
import time
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2

SEQ_LEN = 16
IMG_SIZE = 224

IS_WEBCAM = True

def build_model():
    base_cnn = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # same freezing logic as training
    base_cnn.trainable = True
    for layer in base_cnn.layers[:-40]:
        layer.trainable = False

    model = Sequential([
        TimeDistributed(base_cnn, input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(128),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    return model



# ================= CONFIG =================
MODEL_PATH = r"D:\Machine Learning\camera-surveilance\models\violence_cnn_lstm.keras"
SEQ_LEN = 16
THRESHOLD = 0.6      # lower than before (video-level)
COOLDOWN = 2         # seconds
# =========================================

model = build_model()
model.load_weights(MODEL_PATH)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)


frame_buffer = deque(maxlen=SEQ_LEN)
last_alert = 0

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = preprocess_input(frame.astype("float32"))
    return frame

# 🔁 Choose ONE:
if IS_WEBCAM:
    cap = cv2.VideoCapture(0)   # Webcam
else:
    cap = cv2.VideoCapture("videos/normal/file_000824.avi") # Replace with the video path you want to load with

if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

print("✅ Video surveillance started (press Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess_frame(frame)
    frame_buffer.append(processed)

    display_text = "Warming up..."
    color = (0, 255, 255)

    if len(frame_buffer) == SEQ_LEN:
        clip = np.expand_dims(np.array(frame_buffer), axis=0)  # (1, 16, 224, 224, 3)
        prob = model.predict(clip, verbose=0)[0][0]

        if prob > THRESHOLD:
            display_text = f"ALERT: Violence ({prob*100:.2f}%)"
            color = (0, 0, 255)

            now = time.time()
            if now - last_alert > COOLDOWN:
                winsound.Beep(1200, 300)
                last_alert = now
        else:
            display_text = f"NORMAL ({prob*100:.2f}%)"
            color = (0, 255, 0)

    cv2.putText(
        frame,
        display_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("AI Violence Detection System", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
