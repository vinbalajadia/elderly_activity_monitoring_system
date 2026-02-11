import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # <--- IMPORT THIS

# --- CONFIGURATION ---
MODEL_PATH = "output/activity_model.keras"
CLASSES_PATH = "output/classes.npy"
CONFIDENCE_THRESHOLD = 0.70 

# DISTANCE GATES
MIN_AREA = 15000   
MAX_AREA = 220000 

print("[INFO] Loading system...")
model = tf.keras.models.load_model(MODEL_PATH)
class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}
print(f"[INFO] Classes: {labels}")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("[INFO] System Ready. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. GATEKEEPER
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)

    status_text = "Waiting..."
    status_color = (128, 128, 128)
    box_to_draw = None

    if len(boxes) > 0:
        largest_box = max(boxes, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_box
        area = w * h

        if area < MIN_AREA:
            status_text = "Too Far"
            status_color = (0, 0, 255)
            box_to_draw = largest_box
        elif area > MAX_AREA:
            status_text = "Too Close"
            status_color = (0, 0, 255)
            box_to_draw = largest_box
        else:
            # 2. AI INFERENCE (THE FIX)
            
            # Resize Full Frame (Matches your Padded Training Data)
            input_resized = cv2.resize(frame, (224, 224))
            
            # BGR to RGB
            input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)
            
            # Expand Dims
            input_data = np.expand_dims(input_rgb, axis=0)
            
            # --- APPLY THE MATH FIX ---
            input_data = preprocess_input(input_data)

            # Predict
            predictions = model.predict(input_data, verbose=0)
            class_id = np.argmax(predictions)
            confidence = np.max(predictions)
            label_text = labels[class_id].upper()

            if confidence > CONFIDENCE_THRESHOLD:
                status_text = f"{label_text}: {confidence*100:.1f}%"
                status_color = (0, 255, 0)
            else:
                status_text = f"Uncertain ({confidence*100:.1f}%)"
                status_color = (0, 165, 255)
            
            box_to_draw = largest_box

    # Drawing
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    if box_to_draw is not None:
        bx, by, bw, bh = box_to_draw
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), status_color, 3)

    cv2.imshow("Elderly Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()