import cv2
import numpy as np
import os
from tqdm import tqdm # pip install tqdm if you don't have it, for progress bar

# --- CONFIGURATION ---
INPUT_DATASET = "dataset"          # Your current phone photos
OUTPUT_DATASET = "dataset_fixed"   # Where the new webcam-like photos will go
TARGET_SIZE = (640, 480)           # Standard webcam resolution (4:3 aspect ratio)

def make_square_and_resize(image, target_w, target_h):
    h, w = image.shape[:2]
    target_aspect = target_w / target_h
    current_aspect = w / h

    if current_aspect < target_aspect:
        # Image is too tall (portrait). Pad the sides.
        new_h = h
        new_w = int(new_h * target_aspect)
        pad_horz = (new_w - w) // 2
        pad_vert = 0
    else:
        # Image is too wide (panorama). Pad top/bottom.
        new_w = w
        new_h = int(new_w / target_aspect)
        pad_horz = 0
        pad_vert = (new_h - h) // 2

    # Create black canvas and place image in center
    padded_img = cv2.copyMakeBorder(image, pad_vert, pad_vert, pad_horz, pad_horz, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    # Final resize to standard webcam dimensions
    final_img = cv2.resize(padded_img, (target_w, target_h))
    return final_img

def process_dataset():
    if not os.path.exists(INPUT_DATASET):
        print(f"[ERROR] Input folder '{INPUT_DATASET}' not found.")
        return

    # Get classes (sitting/standing)
    classes = [d for d in os.listdir(INPUT_DATASET) if os.path.isdir(os.path.join(INPUT_DATASET, d))]
    print(f"[INFO] Found classes: {classes}")

    for class_name in classes:
        input_class_dir = os.path.join(INPUT_DATASET, class_name)
        output_class_dir = os.path.join(OUTPUT_DATASET, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        print(f"[INFO] Processing class: {class_name}...")
        images = os.listdir(input_class_dir)
        
        # Process each image with a progress bar
        for img_name in tqdm(images):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_class_dir, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Convert to webcam style
                    fixed_img = make_square_and_resize(img, TARGET_SIZE[0], TARGET_SIZE[1])
                    
                    # Save to new folder
                    output_path = os.path.join(output_class_dir, img_name)
                    cv2.imwrite(output_path, fixed_img)

    print("\n" + "="*40)
    print(f"[SUCCESS] Dataset converted!")
    print(f"New images are in: {OUTPUT_DATASET}/")
    print("="*40)

if __name__ == "__main__":
    process_dataset()