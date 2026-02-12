import cv2
import numpy as np
import time
from flask import Flask, Response
from collections import deque, Counter

# On Raspberry Pi, we often use tflite_runtime.
# If you have full tensorflow installed, change this to: import tensorflow.lite as tflite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("[ERROR] Could not import tflite_runtime or tensorflow.lite")

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "activity_model.tflite" 
CLASSES_PATH = "classes.npy"

# STABILIZER SETTINGS
ENTER_THRESHOLD = 0.70 
EXIT_THRESHOLD = 0.50 
FRAME_BUFFER_SIZE = 8 

# --- STABILIZER CLASS ---
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

# --- INITIALIZATION ---
print("[INFO] Loading Arithmos Pi System...")

# 1. Load Labels
class_names = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_names.items()}

# 2. Load TFLite Model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Initialize HOG (For UI Box Only)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 4. Initialize Stabilizer
stabilizer = ActivityStabilizer(buffer_size=FRAME_BUFFER_SIZE)

# 5. Setup Camera
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

def generate_frames():
    # Initialize FPS variables
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        height, width, _ = frame.shape

        # --- FPS CALCULATION ---
        new_frame_time = time.time()
        fps = 0.0
        if prev_frame_time != 0:
            fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # --- AI INFERENCE (TFLITE) ---
        # 1. Resize
        input_resized = cv2.resize(frame, (224, 224))
        
        # 2. Manual Preprocessing (-1 to 1)
        input_data = input_resized.astype(np.float32)
        input_data = (input_data - 127.5) / 127.5
        input_data = np.expand_dims(input_data, axis=0)

        # 3. Run Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 4. Parse Output
        prediction_idx = np.argmax(output_data)
        raw_confidence = output_data[0][prediction_idx]
        raw_label = labels[prediction_idx].upper()

        # --- STABILIZER ---
        stable_label = stabilizer.update(raw_label, raw_confidence)

        # --- ADAPTIVE UI LOGIC ---
        # Attempt to find person for the box
        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)
        
        largest_box = None
        if len(boxes) > 0:
            largest_box = max(boxes, key=lambda r: r[2] * r[3])

        # DRAWING UI
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
                # --- SCENARIO A: BOX MODE (Good Distance) ---
                x, y, w, h = largest_box
                
                # Draw Box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Label at Top-Left of Box
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - 30), (x + text_w + 10, y), color, -1)
                cv2.putText(frame, display_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            else:
                # --- SCENARIO B: SCREEN MODE (Close Range) ---
                # Thin Screen Border (10px)
                cv2.rectangle(frame, (0, 0), (width, height), color, 10)
                
                # Label at Top-Left of Screen
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(frame, (0, 0), (text_w + 40, 60), color, -1)
                cv2.putText(frame, display_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Draw FPS (Cyan)
        cv2.putText(frame, f"FPS: {int(fps)}", (width - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # ENCODE FOR WEB STREAMING
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # '0.0.0.0' makes it accessible to the Android Phone
    app.run(host='0.0.0.0', port=5000, debug=False)