import cv2
import os

VIDEO_FILENAME = "standing2.mp4" 

CLASS_LABEL = "standing"

VIDEO_PATH = os.path.join("videos", VIDEO_FILENAME)

OUTPUT_DIR = os.path.join("dataset", CLASS_LABEL)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_frames():
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file not found at: {VIDEO_PATH}")
        print("Make sure your video is inside the 'videos' folder.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video {VIDEO_PATH}")
        return

    existing_files = len(os.listdir(OUTPUT_DIR))
    saved_count = existing_files
    
    print(f"--- PROCESSING: {CLASS_LABEL} ---")
    print(f"Reading from: {VIDEO_PATH}")
    print(f"Saving to:    {OUTPUT_DIR}")
    print("\nCONTROLS:")
    print(" [SPACE] - Pause/Play")
    print(" [s]     - Save current frame (Select ROI)")
    print(" [q]     - Quit")
    
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video reached.")
                break

        display_frame = frame.copy()
        
        cv2.putText(display_frame, f"Files Saved: {saved_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if paused:
            cv2.putText(display_frame, "PAUSED - Press 's' to crop", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Video Tagger", display_frame)

        key = cv2.waitKey(30) & 0xFF

        # --- CONTROLS ---
        if key == ord('q'): # Quit
            break
        elif key == ord(' '): # Toggle Pause
            paused = not paused
        elif key == ord('s'): # Save Frame
            # This opens a window to draw a box
            print("\n>> Draw a box around the person. Press ENTER to save. Press 'c' to cancel.")
            
            # Select ROI (Region of Interest)
            roi = cv2.selectROI("Video Tagger", frame, fromCenter=False, showCrosshair=True)
            
            # ROI returns (x, y, w, h). If width or height is 0, user cancelled.
            if roi[2] > 0 and roi[3] > 0:
                x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
                cropped_img = frame[y:y+h, x:x+w]
                
                # Resize to 224x224 (MobileNet Requirement)
                cropped_img = cv2.resize(cropped_img, (224, 224))
                
                # Save file
                filename = f"{CLASS_LABEL}_{saved_count}.jpg"
                save_path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(save_path, cropped_img)
                
                saved_count += 1
                print(f"   Saved: {filename}")
            else:
                print("   Selection cancelled.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_frames()