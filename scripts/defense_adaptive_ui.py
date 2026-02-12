import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input 
from collections import deque, Counter
import time

# --- CONFIGURATION ---
MODEL_PATH = "output/activity_model.keras"
CLASSES_PATH = "output/classes.npy"

# STABILIZER SETTINGS
ENTER_THRESHOLD = 0.70 
EXIT_THRESHOLD = 0.50 
FRAME_BUFFER_SIZE = 8 

class ActivityStabilizer:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        self.current_locked_label = "Waiting..."
        
    def update(self, label, confidence):
        self.buffer.append(label)
        counts = Counter(self.buffer)
        most_common_label, count = counts.most_common(1)[0]
        
        if self.current_locked_label == "Waiting...":
            if confidence > ENTER_THRESHOLD and count > (self.buffer.maxlen // 2):
                self.current_locked_label = label
        else:
            if label == self.current_locked_label:
                if confidence > EXIT_THRESHOLD:
                    self.current_locked_label = label 
                else:
                    self.current_locked_label = "Waiting..."
            else:
                if confidence > ENTER_THRESHOLD and count > (self.buffer.maxlen // 2):
                    self.current_locked_label = most_common_label

        return self.current_locked_label

# --- MAIN SYSTEM ---
print("[INFO] Loading Arithmos Adaptive UI...")
model = tf.keras.models.load_model(MODEL_PATH)
class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}

# VISUAL DETECTOR (HOG) - Controls the UI mode
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

stabilizer = ActivityStabilizer(buffer_size=FRAME_BUFFER_SIZE)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Full Screen Setup
window_name = "Arithmos: Adaptive Monitoring"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret: break
    height, width, _ = frame.shape

    # 1. AI PREDICTION (Always runs on the whole frame)
    input_resized = cv2.resize(frame, (224, 224))
    input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)
    input_data = preprocess_input(np.expand_dims(input_rgb, axis=0))

    predictions = model.predict(input_data, verbose=0)
    class_id = np.argmax(predictions)
    raw_confidence = np.max(predictions)
    raw_label = labels[class_id].upper()

    # 2. STABILIZE RESULT
    stable_label = stabilizer.update(raw_label, raw_confidence)

    # 3. DETECT UI MODE (Box vs Full Screen)
    # We try to find a person for the box
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)
    
    largest_box = None
    if len(boxes) > 0:
        largest_box = max(boxes, key=lambda r: r[2] * r[3])

    # 4. DRAWING THE ADAPTIVE UI
    if stable_label == "Waiting...":
        # GRAY MODE (Scanning)
        color = (100, 100, 100) 
        # Thin Gray Border (Scanning)
        cv2.rectangle(frame, (0, 0), (width, height), color, 4) 
        cv2.putText(frame, "SCANNING AREA...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    else:
        # ACTIVE MODE (Green)
        color = (0, 255, 0)
        display_text = f"{stable_label}: {raw_confidence*100:.0f}%"

        if largest_box is not None:
            # --- SCENARIO A: GOOD ENVIRONMENT (BOX MODE) ---
            x, y, w, h = largest_box
            
            # 1. Draw Box around person
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # 2. Add Label at Top-Left of Box
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - 30), (x + text_w + 10, y), color, -1)
            # Text
            cv2.putText(frame, display_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        else:
            # --- SCENARIO B: CRAMPED/CLOSE (SCREEN MODE) ---
            # 1. Thin Screen Border (10px) - Elegant, not huge
            cv2.rectangle(frame, (0, 0), (width, height), color, 10)
            
            # 2. Label at Top-Left of Screen
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (0, 0), (text_w + 40, 60), color, -1)
            # Text
            cv2.putText(frame, display_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()