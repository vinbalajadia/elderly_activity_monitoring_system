import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input 
from collections import deque, Counter

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
        
        # Sticky Logic
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
print("[INFO] Loading Arithmos Final System...")
model = tf.keras.models.load_model(MODEL_PATH)
class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}

# VISUAL DETECTOR (For the Box only)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

stabilizer = ActivityStabilizer(buffer_size=FRAME_BUFFER_SIZE)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Full Screen (Optional)
window_name = "Arithmos Live Demo"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret: break

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

    # 3. FIND BOX (For UI Only)
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)
    
    largest_box = None
    if len(boxes) > 0:
        largest_box = max(boxes, key=lambda r: r[2] * r[3])

    # 4. DRAWING UI
    if stable_label != "Waiting...":
        color = (0, 255, 0) # Green for Active
        text_display = f"{stable_label}: {raw_confidence*100:.1f}%"
        
        if largest_box is not None:
            # SCENARIO A: Box Found (Mid-Distance)
            x, y, w, h = largest_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Label background
            (text_w, text_h), _ = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - 35), (x + text_w, y), color, -1)
            cv2.putText(frame, text_display, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            # SCENARIO B: No Box (Too Close / Full Screen)
            # Draw a thick Green Border around the whole screen
            cv2.rectangle(frame, (0, 0), (640, 480), color, 15) 
            
            # Text at Top Left
            cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
            cv2.putText(frame, "CLOSE RANGE: " + text_display, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    else:
        # Waiting State
        cv2.putText(frame, "Scanning...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()