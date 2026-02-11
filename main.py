import cv2
import numpy as np
import tensorflow as tf
import time  # <--- NEW IMPORT FOR FPS
from flask import Flask, Response

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "activity_model.tflite"
CLASSES_PATH = "classes.npy"
CONFIDENCE_THRESHOLD = 0.70 
MIN_AREA = 15000  
MAX_AREA = 220000 

# Load Labels
class_names = np.load(CLASSES_PATH, allow_pickle=True).item()
idx_to_label = {v: k for k, v in class_names.items()}

# Load TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)
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

        # --- FPS CALCULATION ---
        new_frame_time = time.time()
        # Calculate FPS (1 / time_diff)
        # We use a safe default of 0.0 to prevent division by zero on the first frame
        fps = 0.0
        if prev_frame_time != 0:
            fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # -----------------------

        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)
        status_text = "Waiting..."
        status_color = (100, 100, 100)
        final_box = None 

        if len(boxes) > 0:
            largest_box = max(boxes, key=lambda r: r[2] * r[3])
            x, y, w, h = largest_box
            area = w * h

            if area < MIN_AREA:
                status_text = "Camera Too Far"
                status_color = (0, 0, 255)
            elif area > MAX_AREA:
                status_text = "Camera Too Close"
                status_color = (0, 0, 255)
            else:
                # --- AI INFERENCE ---
                input_resized = cv2.resize(frame, (224, 224))
                input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)
                input_data = np.expand_dims(input_rgb, axis=0)
                
                # Manual Preprocessing (0-255 to -1 to 1)
                input_data = (np.float32(input_data) - 127.5) / 127.5

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                prediction_idx = np.argmax(output_data)
                confidence = output_data[0][prediction_idx]
                label = idx_to_label[prediction_idx]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    status_text = f"{label.upper()}: {confidence*100:.1f}%"
                    status_color = (0, 255, 0)
                    final_box = (x, y, w, h)
                else:
                    status_text = "Uncertain"
                    status_color = (0, 165, 255)

        # --- DRAW UI ---
        # 1. Black Top Bar
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        
        # 2. Status Text (Left Side)
        cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # 3. FPS Counter (Right Side)
        # Drawn in Cyan (255, 255, 0)
        cv2.putText(frame, f"FPS: {int(fps)}", (480, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 4. Bounding Box
        if final_box is not None:
            fx, fy, fw, fh = final_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Using '0.0.0.0' allows you to access this stream from other computers on the network
    app.run(host='0.0.0.0', port=5000, debug=False)