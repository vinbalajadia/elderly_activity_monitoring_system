import cv2
import numpy as np
import tensorflow as tf
import time

# --- CONFIGURATION ---
MODEL_PATH = "activity_model.tflite"
CLASSES_PATH = "classes.npy"
CONFIDENCE_THRESHOLD = 0.70  # Only show box if 70% sure

# --- DISTANCE SETTINGS (Adjust for your room) ---
# Screen Area = 640 * 480 = 307,200 pixels
MIN_AREA = 15000   # Below this = "Too Far"
MAX_AREA = 220000  # Above this = "Too Close"

# Load Labels
class_names = np.load(CLASSES_PATH, allow_pickle=True).item()
idx_to_label = {v: k for k, v in class_names.items()}

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- INITIALIZE PERSON DETECTOR (The Gatekeeper) ---
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("[INFO] System Ready. Starting Camera...")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. RUN PERSON DETECTOR (HOG)
    # This finds the person so we can measure distance & draw the box
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)

    status_text = "Waiting..."
    status_color = (100, 100, 100) # Gray
    
    # Variables to hold the final box coordinates
    final_box = None 

    if len(boxes) > 0:
        # Find the largest person (The main subject)
        largest_box = max(boxes, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_box
        area = w * h

        # --- GATE 2: CHECK DISTANCE ---
        if area < MIN_AREA:
            status_text = "Camera Too Far"
            status_color = (0, 0, 255) # Red (No Box)
        
        elif area > MAX_AREA:
            status_text = "Camera Too Close"
            status_color = (0, 0, 255) # Red (No Box)
            
        else:
            # --- GATE 3: RUN AI (Distance is Good) ---
            
            # PREPROCESSING (The "Squish" Method)
            # We ignore the HOG box for the AI input and use the center strip
            # This ensures it matches your training data perfectly.
            height, width, _ = frame.shape
            new_width = 270 
            start_x = (width - new_width) // 2
            
            # Crop Center Strip -> Resize to 224x224
            cropped_frame = frame[:, start_x : start_x + new_width]
            input_img = cv2.resize(cropped_frame, (224, 224))
            
            # Normalize
            input_data = np.expand_dims(input_img, axis=0)
            input_data = (np.float32(input_data) / 255.0)

            # INFERENCE
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            prediction_idx = np.argmax(output_data)
            confidence = output_data[0][prediction_idx]
            label = idx_to_label[prediction_idx]
            
            if confidence > CONFIDENCE_THRESHOLD:
                # SUCCESS! We found a valid activity.
                status_text = f"{label.upper()}: {confidence*100:.1f}%"
                status_color = (0, 255, 0) # Green Text
                
                # Save the box coordinates to draw later
                final_box = (x, y, w, h)
            else:
                status_text = "Uncertain"
                status_color = (0, 165, 255) # Orange (No Box)

    # --- DRAWING PHASE ---
    
    # 1. Draw the Status Bar (Top)
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # 2. Draw the Green Box (ONLY if activity detected)
    if final_box is not None:
        fx, fy, fw, fh = final_box
        # Draw Green Box
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)
        # Optional: Draw label above box
        cv2.putText(frame, status_text.split(":")[0], (fx, fy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Elderly Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()