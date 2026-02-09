import cv2
import numpy as np
import tensorflow as tf
import os

# --- CONFIGURATION ---
MODEL_PATH = "output/activity_model.keras"  # PC uses the Keras model directly
CLASSES_PATH = "output/classes.npy"
VIDEO_SOURCE = 0  # IV Cam usually shows up as 0 or 1

# --- DISTANCE SETTINGS (Same as Pi) ---
# Adjust these if your PC webcam is different from the Pi Camera
MIN_AREA = 15000   # Below this = "Too Far"
MAX_AREA = 220000  # Above this = "Too Close"
CONFIDENCE_THRESHOLD = 0.70

# Load the trained model
print("[INFO] Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}
print(f"[INFO] Classes: {labels}")

# Initialize "Gatekeeper" (Person Detector)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(VIDEO_SOURCE)

# Set resolution to match Pi (helps consistency)
cap.set(3, 640)
cap.set(4, 480)

print("[INFO] Starting Test...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. RUN PERSON DETECTOR (HOG)
    # This simulates the "Gatekeeper" logic on the Pi
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)

    status_text = "Waiting..."
    status_color = (100, 100, 100) # Gray
    final_box = None

    if len(boxes) > 0:
        # Find the largest person
        largest_box = max(boxes, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_box
        area = w * h

        # --- GATE 2: CHECK DISTANCE ---
        if area < MIN_AREA:
            status_text = "Camera Too Far"
            status_color = (0, 0, 255) # Red
        elif area > MAX_AREA:
            status_text = "Camera Too Close"
            status_color = (0, 0, 255) # Red
        else:
            # --- GATE 3: RUN AI (The "Squish" Logic) ---
            
            # CRITICAL: We must crop the center strip to mimic the vertical phone video
            height, width, _ = frame.shape
            new_width = 270 
            start_x = (width - new_width) // 2
            
            # Crop Center Strip -> Resize to 224x224 (Squish)
            cropped_frame = frame[:, start_x : start_x + new_width]
            resized_frame = cv2.resize(cropped_frame, (224, 224))
            
            # Normalize
            input_data = resized_frame / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            # Predict
            predictions = model.predict(input_data, verbose=0)
            class_id = np.argmax(predictions)
            confidence = np.max(predictions)
            label = labels[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                status_text = f"{label.upper()}: {confidence*100:.1f}%"
                status_color = (0, 255, 0) # Green
                final_box = (x, y, w, h)   # Save box to draw later
            else:
                status_text = "Uncertain"
                status_color = (0, 165, 255) # Orange

            # Debug: Show the "Squished" view in corner
            # debug_view = cv2.resize(resized_frame, (100, 100))
            # frame[0:100, 0:100] = (debug_view * 255).astype(np.uint8)

    # --- DRAWING ---
    # 1. Status Bar
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # 2. Green Box (Only if Activity Detected)
    if final_box is not None:
        fx, fy, fw, fh = final_box
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)

    cv2.imshow("Activity Monitor Test (PC Version)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()