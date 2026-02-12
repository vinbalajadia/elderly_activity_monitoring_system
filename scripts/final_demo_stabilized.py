import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input 
from collections import deque, Counter

# --- CONFIGURATION ---
MODEL_PATH = "output/activity_model.keras"
CLASSES_PATH = "output/classes.npy"

# --- TUNING KNOBS (Adjust these to make it less "tiring") ---
# 1. How sure the AI needs to be to START detecting (0.70 = 70%)
ENTER_THRESHOLD = 0.70 

# 2. How sure the AI needs to be to KEEP detecting (0.50 = 50%)
# This "Sticky" threshold prevents the label from disappearing quickly.
EXIT_THRESHOLD = 0.50 

# 3. How many frames to remember (Smoothing)
# Higher = More stable but slower reaction. Lower = Faster but flickers.
FRAME_BUFFER_SIZE = 8 

class ActivityStabilizer:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        self.current_locked_label = "Waiting..."
        self.current_confidence = 0.0

    def update(self, label, confidence):
        # Add new prediction to history
        self.buffer.append(label)

        # 1. VOTING: What is the most common label in the last N frames?
        counts = Counter(self.buffer)
        most_common_label, count = counts.most_common(1)[0]
        
        # 2. HYSTERESIS (Sticky Logic)
        if self.current_locked_label == "Waiting...":
            # Harder to enter a state
            if confidence > ENTER_THRESHOLD and count > (self.buffer.maxlen // 2):
                self.current_locked_label = label
        else:
            # Easier to stay in a state
            # If the new label matches the locked one, we accept lower confidence
            if label == self.current_locked_label:
                if confidence > EXIT_THRESHOLD:
                    self.current_locked_label = label # Keep it
                else:
                    self.current_locked_label = "Waiting..." # Lost it
            else:
                # If the AI strongly disagrees with the locked label, switch
                if confidence > ENTER_THRESHOLD and count > (self.buffer.maxlen // 2):
                    self.current_locked_label = most_common_label

        return self.current_locked_label

# --- MAIN SYSTEM ---
print("[INFO] Loading Arithmos System...")
model = tf.keras.models.load_model(MODEL_PATH)
class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}
print(f"[INFO] Classes: {labels}")

# Initialize Stabilizer
stabilizer = ActivityStabilizer(buffer_size=FRAME_BUFFER_SIZE)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Full Screen Setup (Optional - Uncomment to use)
# window_name = "Arithmos Stabilized"
# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("[INFO] System Ready. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. PRE-PROCESSING
    # No HOG Gatekeeper anymore - we send the whole frame to AI
    # This ensures we get a prediction even if HOG fails
    
    # Resize to standard 224x224
    input_resized = cv2.resize(frame, (224, 224))
    input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(input_rgb, axis=0)
    
    # MATHEMATICAL FIX (-1 to 1 scale)
    input_data = preprocess_input(input_data)

    # 2. INFERENCE
    predictions = model.predict(input_data, verbose=0)
    class_id = np.argmax(predictions)
    raw_confidence = np.max(predictions)
    raw_label = labels[class_id].upper()

    # 3. STABILIZATION ENGINE
    stable_label = stabilizer.update(raw_label, raw_confidence)

    # 4. DRAWING UI
    # Dynamic Color: Green if Active, Grey if Waiting
    if stable_label != "Waiting...":
        status_color = (0, 255, 0) # Green
        status_text = f"{stable_label}"
    else:
        status_color = (128, 128, 128) # Grey
        status_text = "Scanning..."

    # Draw Top Bar
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    
    # Text
    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Debug Info (Small text at bottom)
    debug_text = f"Raw: {raw_label} ({raw_confidence*100:.0f}%)"
    cv2.putText(frame, debug_text, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Arithmos Stabilized", frame)
    # cv2.imshow(window_name, frame) # Use this if full screen

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()