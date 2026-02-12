import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input 
from collections import deque, Counter

# --- CONFIGURATION ---
MODEL_PATH = "output/activity_model.keras"
CLASSES_PATH = "output/classes.npy"

# STABILIZER SETTINGS
# 0.70 to start detecting, 0.50 to keep detecting (The "Sticky" Logic)
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
        
        # Sticky Logic (Prevents flickering)
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
print("[INFO] Loading Arithmos Defense System...")
model = tf.keras.models.load_model(MODEL_PATH)
class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}

stabilizer = ActivityStabilizer(buffer_size=FRAME_BUFFER_SIZE)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Full Screen Setup
window_name = "Arithmos: Elderly Activity Monitoring"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    height, width, _ = frame.shape

    # 1. AI PREDICTION
    input_resized = cv2.resize(frame, (224, 224))
    input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)
    input_data = preprocess_input(np.expand_dims(input_rgb, axis=0))

    predictions = model.predict(input_data, verbose=0)
    class_id = np.argmax(predictions)
    raw_confidence = np.max(predictions)
    raw_label = labels[class_id].upper()

    # 2. STABILIZE RESULT
    stable_label = stabilizer.update(raw_label, raw_confidence)

    # 3. DRAWING THE UI
    if stable_label == "Waiting...":
        # GRAY MODE (Scanning)
        border_color = (100, 100, 100) 
        main_text = "SCANNING..."
        confidence_text = "Analysis pending..."
        bar_length = 0
    else:
        # GREEN MODE (Detected)
        border_color = (0, 255, 0) 
        main_text = stable_label
        confidence_text = f"CONFIDENCE: {raw_confidence*100:.1f}%"
        bar_length = int(raw_confidence * 200) # Scale to 200px width

    # Draw Thick Border
    cv2.rectangle(frame, (0, 0), (width, height), border_color, 20)
    
    # Draw Top Text Panel
    cv2.rectangle(frame, (0, 0), (width, 80), border_color, -1)
    
    # Draw Main Activity Text (Black text for contrast)
    cv2.putText(frame, main_text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # --- CONFIDENCE BAR (Bottom Right) ---
    if stable_label != "Waiting...":
        # Background of bar
        bar_x = width - 250
        bar_y = 45
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 200, bar_y + 15), (50, 50, 50), -1)
        
        # Filled part of bar (White fill looks good on Green/Gray)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_length, bar_y + 15), (255, 255, 255), -1)
        
        # Text Label above bar
        cv2.putText(frame, confidence_text, (bar_x, bar_y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()