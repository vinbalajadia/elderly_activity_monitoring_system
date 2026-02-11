import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # <--- CRITICAL IMPORT

# --- CONFIGURATION ---
MODEL_PATH = "output/activity_model.keras"
CLASSES_PATH = "output/classes.npy"
TEST_IMAGE_PATH = "dataset/standing/standing_0.jpg" # The image you just tested

def make_square_with_padding(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    target_w, target_h = 640, 480 
    
    scale = min(target_w/w, target_h/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    final_input = cv2.resize(padded, target_size)
    return final_input

def test_with_visual_result():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found.")
        return

    # Load Model
    print(f"[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
    labels = {v: k for k, v in class_indices.items()}

    # Load Image
    frame = cv2.imread(TEST_IMAGE_PATH)
    if frame is None:
        print("[ERROR] Could not load image.")
        return

    # Pad & Resize
    input_img = make_square_with_padding(frame)
    
    # Preprocess
    input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    # --- THE CRITICAL FIX IS HERE ---
    # OLD WAY (WRONG): input_data = input_rgb / 255.0
    # NEW WAY (CORRECT): Use the MobileNet preprocessor
    
    # We must expand dims FIRST, then preprocess
    input_data = np.expand_dims(input_rgb, axis=0)
    
    # This function handles the -1 to 1 scaling automatically
    input_data = preprocess_input(input_data) 

    # Predict
    predictions = model.predict(input_data, verbose=0)
    class_id = np.argmax(predictions)
    label_text = labels[class_id].upper()
    confidence = np.max(predictions)

    # --- VISUALIZE RESULT ---
    result_text = f"{label_text}: {confidence*100:.1f}%"
    color = (0, 255, 0) if label_text == "STANDING" else (0, 0, 255)
    
    print(f"FINAL VERDICT: {label_text} ({confidence*100:.1f}%)")
    
    display_img = cv2.resize(input_img, (500, 500))
    cv2.putText(display_img, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Test Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_with_visual_result()