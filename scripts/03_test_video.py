# scripts/03_test_video.py

import cv2
import numpy as np
import tensorflow as tf
import os

# --- CONFIGURATION ---
MODEL_PATH = "output/activity_model.keras"
CLASSES_PATH = "output/classes.npy"

# Use 0 for Webcam, or put a filename like "videos/test_video.mp4"
VIDEO_SOURCE = 0 

# Load the trained model
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
# Flip it so we have {0: "getting_up", 1: "resting"...}
labels = {v: k for k, v in class_indices.items()}
print(f"[INFO] Classes: {labels}")

cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Prepare the image for the model
    # MobileNet expects 224x224
    resized_frame = cv2.resize(frame, (224, 224))
    
    # Normalize (0-1) just like in training
    input_data = resized_frame / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    input_data = np.expand_dims(input_data, axis=0)

    # 2. Predict
    predictions = model.predict(input_data, verbose=0)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)
    label = labels[class_id]

    # 3. Draw on screen
    # Color: Green if high confidence, Red if low
    color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
    
    text = f"{label}: {confidence*100:.1f}%"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # --- REMOVED "BRAIN VIEW" CODE BELOW ---
    # small_view = cv2.resize(resized_frame, (100, 100))
    # frame[0:100, 0:100] = small_view
    # ---------------------------------------

    cv2.imshow("Activity Monitor Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()