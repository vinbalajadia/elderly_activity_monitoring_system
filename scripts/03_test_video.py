import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = "output/activity_model.keras"
CLASSES_PATH = "output/classes.npy"

# DISTANCE GATES (Adjust these if needed!)
MIN_AREA = 15000   # If box is smaller than this -> "Too Far"
MAX_AREA = 220000  # If box is bigger than this -> "Too Close"

# CONFIDENCE THRESHOLD
CONFIDENCE_THRESHOLD = 0.70 # Only show Sit/Stand if AI is 70% sure

# --- LOAD MODEL & CLASSES ---
print("[INFO] Loading system...")
model = tf.keras.models.load_model(MODEL_PATH)
class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}
print(f"[INFO] Classes: {labels}")

# --- INITIALIZE GATEKEEPER (HOG DETECTOR) ---
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --- START VIDEO ---
cap = cv2.VideoCapture(0) # 0 = Webcam
cap.set(3, 640)
cap.set(4, 480)

print("[INFO] System Ready. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. RUN GATEKEEPER (Detect Person)
    # scale=1.1 means "look for people 10% bigger in each pass" (Standard speed/accuracy balance)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.1)

    status_text = "Waiting..."
    status_color = (128, 128, 128) # Gray
    box_to_draw = None

    if len(boxes) > 0:
        # Get the largest person (Main User)
        largest_box = max(boxes, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_box
        area = w * h

        # 2. CHECK DISTANCE
        if area < MIN_AREA:
            status_text = "Too Far - Move Closer"
            status_color = (0, 0, 255) # Red
            box_to_draw = largest_box
        elif area > MAX_AREA:
            status_text = "Too Close - Move Back"
            status_color = (0, 0, 255) # Red
            box_to_draw = largest_box
        else:
            # 3. RUN AI (Only if distance is good)
            
            # Crop the person out
            person_roi = frame[y:y+h, x:x+w]
            
            # Preprocess for AI (Resize -> RGB -> Normalize)
            try:
                roi_resized = cv2.resize(person_roi, (224, 224))
                roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB) # Fix Blue Man
                input_data = roi_rgb / 255.0
                input_data = np.expand_dims(input_data, axis=0)

                # Predict
                predictions = model.predict(input_data, verbose=0)
                class_id = np.argmax(predictions)
                confidence = np.max(predictions)
                label_text = labels[class_id].upper()

                if confidence > CONFIDENCE_THRESHOLD:
                    status_text = f"{label_text}: {confidence*100:.1f}%"
                    status_color = (0, 255, 0) # Green
                else:
                    status_text = f"Uncertain ({confidence*100:.1f}%)"
                    status_color = (0, 165, 255) # Orange
                
                box_to_draw = largest_box

            except Exception as e:
                print(f"Error processing ROI: {e}")

    # --- DRAWING ---
    # Top Bar Background
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    
    # Status Text
    cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Draw Box around Person (if found)
    if box_to_draw is not None:
        bx, by, bw, bh = box_to_draw
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), status_color, 3)

    cv2.imshow("Elderly Monitoring System (Final)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()