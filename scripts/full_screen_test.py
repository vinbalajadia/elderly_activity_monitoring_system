import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input 

# --- CONFIGURATION ---
MODEL_PATH = "output/activity_model.keras"
CLASSES_PATH = "output/classes.npy"
CONFIDENCE_THRESHOLD = 0.70 

# DISTANCE GATES (Based on Chapter 3 Methodology)
MIN_AREA = 15000   
MAX_AREA = 220000 

print("[INFO] Loading Arithmos System...")
model = tf.keras.models.load_model(MODEL_PATH)
class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}
print(f"[INFO] Classes detected: {labels}")

# Initialize Person Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- FULL SCREEN SETUP ---
window_name = "Arithmos: Real-Time Elderly Monitoring"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("[INFO] System Ready. Stand in front of the camera.")
print("[INFO] PRESS 'Q' TO EXIT FULL SCREEN.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. GATEKEEPER (HOG Detection)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)

    status_text = "Waiting..."
    status_color = (128, 128, 128) # Grey
    box_to_draw = None

    if len(boxes) > 0:
        # Focus on the largest person in the frame
        largest_box = max(boxes, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_box
        area = w * h

        # Check Distance Gates
        if area < MIN_AREA:
            status_text = "Status: Too Far"
            status_color = (0, 0, 255) # Red
            box_to_draw = largest_box
        elif area > MAX_AREA:
            status_text = "Status: Too Close"
            status_color = (0, 0, 255) # Red
            box_to_draw = largest_box
        else:
            # 2. AI INFERENCE (The Geometry Fix)
            # Resize 4:3 frame to 224x224 square for MobileNet
            input_resized = cv2.resize(frame, (224, 224))
            input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(input_rgb, axis=0)
            
            # Apply MobileNetV3 Normalization (-1 to 1)
            input_data = preprocess_input(input_data)

            # Perform Classification
            predictions = model.predict(input_data, verbose=0)
            class_id = np.argmax(predictions)
            confidence = np.max(predictions)
            label_text = labels[class_id].upper()

            if confidence > CONFIDENCE_THRESHOLD:
                status_text = f"{label_text} ({confidence*100:.1f}%)"
                status_color = (0, 255, 0) # Green
            else:
                status_text = "Analyzing..."
                status_color = (0, 165, 255) # Orange
            
            box_to_draw = largest_box

    # --- DRAWING THE UI ---
    # Top Information Bar
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Bottom Instruction
    cv2.putText(frame, "Press 'Q' to Quit", (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Bounding Box
    if box_to_draw is not None:
        bx, by, bw, bh = box_to_draw
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), status_color, 3)

    # Show Frame
    cv2.imshow(window_name, frame)

    # Quit logic
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()