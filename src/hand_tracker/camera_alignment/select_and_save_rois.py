import cv2
import json
import os

def select_and_save_rois(video_files, output_file="camera_rois.json"):
    """
    Opens the first frame of each video, lets user select ROI, 
    and saves the coordinates to a JSON file.
    """
    roi_dict = {}

    for cam_name, video_path in video_files.items():
        print(f"Opening {cam_name}...")
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Error: Could not read video for {cam_name}")
            continue

        # Instructions
        print(f"Select ROI for {cam_name}. Press SPACE/ENTER to confirm. Press 'c' to cancel.")
        
        # Opens the GUI to draw the box
        # Returns tuple (x, y, w, h)
        roi = cv2.selectROI(f"Select ROI for {cam_name}", frame, showCrosshair=True)
        cv2.destroyWindow(f"Select ROI for {cam_name}")
        
        # Check if user cancelled (all zeros)
        if roi == (0, 0, 0, 0):
            print(f"Skipped {cam_name}")
            continue

        # Store in dictionary
        roi_dict[cam_name] = roi
        print(f"Saved {cam_name}: {roi}")

    # Save to JSON file for later use
    with open(output_file, 'w') as f:
        json.dump(roi_dict, f, indent=4)
    
    print(f"\nSuccess! ROIs saved to {output_file}")

# --- USAGE ---

# 1. Define camera inputs (Day 1 / Baseline videos)
video_folder_dir = "/media/yiting/NewVolume/Data/Videos/Camera_Alignment/2025-12-18/cameras/2025-12-18_11-22-05_062292"
cameras = {
    "camTo": os.path.join(video_folder_dir, "camTo-orig.mp4"),
    "camTL": os.path.join(video_folder_dir, "camTL-orig.mp4"),
    "camTR": os.path.join(video_folder_dir, "camTR-orig.mp4"),
    "camBL": os.path.join(video_folder_dir, "camBL-orig.mp4"),
    "camBR": os.path.join(video_folder_dir, "camBR-orig.mp4")
}
output_file_path = "/media/yiting/NewVolume/Data/Videos/Camera_Alignment/ROIs/2025-12-18_camera_rois.json"

# 2. Run the selector
select_and_save_rois(cameras, output_file=output_file_path)