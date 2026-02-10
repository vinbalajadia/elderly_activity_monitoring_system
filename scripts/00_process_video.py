import cv2
import os

# --- CONFIGURATION ---
VIDEO_FILENAME = "sitting15.mp4" 
CLASS_LABEL = "sitting"

# Paths
VIDEO_PATH = os.path.join("videos", VIDEO_FILENAME)
OUTPUT_DIR = os.path.join("dataset", CLASS_LABEL)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_frames():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file not found at: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video {VIDEO_PATH}")
        return

    existing_files = len(os.listdir(OUTPUT_DIR))
    saved_count = existing_files
    
    print(f"--- PROCESSING: {CLASS_LABEL} ---")
    print("Press [s] to save. The image will look 'SQUISHED'. This is OKAY.")
    
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video reached.")
                break
        
        # --- 1. FORCE RESIZE TO 224x224 (The "Squish") ---
        # We do this immediately so what you save is what you get.
        # The person will look thin/tall. The AI can learn this!
        ai_input_frame = cv2.resize(frame, (224, 224))
        
        # --- 2. CREATE DISPLAY FRAME (For your eyes only) ---
        # We zoom this up so you can see it clearly on your screen
        display_frame = cv2.resize(ai_input_frame, (448, 448))
        
        # Draw status
        cv2.putText(display_frame, f"Saved: {saved_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if paused:
            cv2.putText(display_frame, "PAUSED", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Video Tagger (Squished View)", display_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            # Save the 224x224 "Squished" frame
            filename = f"{CLASS_LABEL}_{saved_count}.jpg"
            save_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(save_path, ai_input_frame)
            
            saved_count += 1
            print(f"   [SAVED] {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_frames()