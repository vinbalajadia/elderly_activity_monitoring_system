import tensorflow as tf
import os

# Point to the NEW .keras file
MODEL_PATH = "output/activity_model.keras"
TFLITE_PATH = "output/activity_model.tflite"

def convert_model():
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File not found. Did you run the training script?")
        return

    # Load Keras model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return

    # Convert
    print("[INFO] Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    
    print(f"[SUCCESS] Saved to {TFLITE_PATH}")

if __name__ == "__main__":
    convert_model()