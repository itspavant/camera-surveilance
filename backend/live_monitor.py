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

IS_WEBCAM = False   # set False when using video file

INFER_EVERY_N_FRAMES_WEBCAM = 2
INFER_EVERY_N_FRAMES_VIDEO = 5

VIDEO_DELAY_MS = 30   # smooth playback for files (~33ms = 30fps)

frame_count = 0
last_prob = None


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
MODEL_PATH = r"PATH\models\violence_cnn_lstm.keras" # Ensure to give path to model
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

if IS_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("videos/violent/file_000004.avi")


if not cap.isOpened():
    print("❌ Cannot open video source")
    exit()

print("✅ Video surveillance started (press Q to quit)")

frame_count = 0
last_prob = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Preprocess and buffer frames (VIDEO, not frame-based)
    processed = preprocess_frame(frame)
    frame_buffer.append(processed)

    display_text = "Monitoring..."
    color = (255, 255, 255)

    # Choose inference rate based on input source
    infer_rate = (
        INFER_EVERY_N_FRAMES_WEBCAM
        if IS_WEBCAM
        else INFER_EVERY_N_FRAMES_VIDEO
    )

    # Run inference ONLY every N frames
    if len(frame_buffer) == SEQ_LEN and frame_count % infer_rate == 0:
        clip = np.expand_dims(np.array(frame_buffer), axis=0)
        last_prob = model.predict(clip, verbose=0)[0][0]

    # Use last prediction for display
    if last_prob is not None:
        if last_prob > THRESHOLD:
            display_text = f"ALERT: Violence Detected ({last_prob*100:.1f}%)"
            color = (0, 0, 255)

            now = time.time()
            if now - last_alert > COOLDOWN:
                winsound.Beep(1200, 300)
                last_alert = now
        else:
            display_text = f"Normal Activity ({last_prob*100:.1f}%)"
            color = (0, 255, 0)

    # Overlay result
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

    # Proper pacing + quit on 'q'
    if IS_WEBCAM:
        key = cv2.waitKey(1) & 0xFF
    else:
        key = cv2.waitKey(VIDEO_DELAY_MS) & 0xFF

    if key == ord('q'):
        print("🛑 Exiting surveillance...")
        break




cap.release()
cv2.destroyAllWindows()
