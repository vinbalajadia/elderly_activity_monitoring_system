import cv2
import os
import shutil
import numpy as np

# --- CONFIGURATION ---
INPUT_DIR = "dataset"          # Your current images
OUTPUT_DIR = "dataset_motion"  # Where the new "ghost" images will go

# Create the new folder (Delete old one if it exists to start fresh)
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print(f"[INFO] Processing images from {INPUT_DIR} to {OUTPUT_DIR}...")

# Categories to process
categories = ["sitting", "standing"]

for category in categories:
    input_path = os.path.join(INPUT_DIR, category)
    output_path = os.path.join(OUTPUT_DIR, category)
    
    if not os.path.exists(input_path):
        print(f"[WARN] Folder not found: {input_path}")
        continue
        
    os.makedirs(output_path, exist_ok=True)
    
    # Get all images and SORT them (Crucial! Time must flow forward)
    # We sort by filename number (assuming format like sitting_10.jpg)
    try:
        images = sorted(os.listdir(input_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
    except:
        # Fallback if filenames are weird
        images = sorted(os.listdir(input_path))

    prev_frame = None
    count = 0

    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        
        # 1. Read Image
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # 2. Convert to Grayscale (Motion doesn't need color)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 3. Apply Gaussian Blur (Reduces camera noise/static)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue # Skip the very first frame

        # 4. Calculate Difference (Current - Previous)
        frame_diff = cv2.absdiff(prev_frame, gray)
        
        # 5. Threshold (Optional: Removes tiny shadows)
        # _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # 6. Convert back to BGR (MobileNet expects 3 channels)
        # We stack the grayscale difference 3 times
        motion_img = cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)

        # 7. Save
        cv2.imwrite(os.path.join(output_path, img_name), motion_img)
        
        # Update previous frame
        prev_frame = gray
        count += 1

    print(f"[SUCCESS] Processed {count} images for {category}")

print("[INFO] Done. You can now train on 'dataset_motion'.")