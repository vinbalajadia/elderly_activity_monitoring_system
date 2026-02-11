import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response

# --- FLASK APP ---
app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "activity_model.tflite"
CLASSES_PATH = "classes.npy"
CONFIDENCE_THRESHOLD = 0.70 

# DISTANCE SETTINGS
MIN_AREA = 15000  
MAX_AREA = 220000 

# Load Labels
class_names = np.load(CLASSES_PATH, allow_pickle=True).item()
idx_to_label = {v: k for k, v in class_names.items()}

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize Person Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. RUN PERSON DETECTOR (HOG)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)

        status_text = "Waiting..."
        status_color = (100, 100, 100) # Gray
        final_box = None 

        if len(boxes) > 0:
            largest_box = max(boxes, key=lambda r: r[2] * r[3])
            x, y, w, h = largest_box
            area = w * h

            if area < MIN_AREA:
                status_text = "Camera Too Far"
                status_color = (0, 0, 255) # Red
            elif area > MAX_AREA:
                status_text = "Camera Too Close"
                status_color = (0, 0, 255) # Red
            else:
                # --- AI INFERENCE (THE FIX) ---
                # We use the WHOLE frame (No cropping!)
                
                # 1. Resize to 224x224
                input_resized = cv2.resize(frame, (224, 224))
                
                # 2. Convert BGR to RGB (Fix the "Blue Man" bug)
                input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)
                
                # 3. Normalize (0 to 1) & Add Batch Dimension
                input_data = np.expand_dims(input_rgb, axis=0)
                input_data = (np.float32(input_data) / 255.0)

                # 4. Run TFLite
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                prediction_idx = np.argmax(output_data)
                confidence = output_data[0][prediction_idx]
                label = idx_to_label[prediction_idx]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    status_text = f"{label.upper()}: {confidence*100:.1f}%"
                    status_color = (0, 255, 0) # Green
                    final_box = (x, y, w, h)
                else:
                    status_text = "Uncertain"
                    status_color = (0, 165, 255) # Orange

        # --- DRAW UI ---
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        if final_box is not None:
            fx, fy, fw, fh = final_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)
            # Optional: Label above box
            # cv2.putText(frame, status_text.split(":")[0], (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # --- ENCODE FRAME TO JPEG FOR STREAMING ---
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Elderly Monitoring System Running</h1><p>Stream is at <a href='/video_feed'>/video_feed</a></p>"

if __name__ == '__main__':
    # host='0.0.0.0' allows access from other devices on the network
    app.run(host='0.0.0.0', port=5000, debug=False)