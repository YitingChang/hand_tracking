import cv2
import glob
import os
import numpy as np
import json
import argparse
import sys

# --- Constants (Defaults) ---
DEFAULT_ROIS_PATH = '/media/yiting/NewVolume/Data/Videos/Camera_Alignment/ROIs/2025-12-18_camera_rois.json'
DEFAULT_BASE_DIR = '/media/yiting/NewVolume/Data/Videos/Camera_Alignment/2025-12-18/cameras/2025-12-18_11-22-05_062292'
CAMERAS_LIST = ['camTo', 'camTL', 'camTR', 'camBL', 'camBR'] 

def get_single_frame(vid_path, frame_idx=10):
    '''
    Extract a single frame from a video
    '''
    if not os.path.exists(vid_path):
        print(f"Warning: File not found -> {vid_path}")
        return None

    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Warning: Could not read frame {frame_idx} from {os.path.basename(vid_path)}")
        return None

    return frame

def load_rois(json_path):
    if not os.path.exists(json_path):
        print(f"Error: ROI file not found at {json_path}")
        return {}
        
    with open(json_path, 'r') as f:
        roi_data = json.load(f)
    
    return roi_data

def quantify_camera_shift(test_img, template, x_base, y_base):
    ''' 
    Calculates the pixel shift of the template within the test_img.
    '''
    # Template Matching (TM_CCOEFF_NORMED is best for lighting invariance)
    res = cv2.matchTemplate(test_img, template, cv2.TM_CCOEFF_NORMED)
    
    # Get the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # max_loc is the top-left corner (x, y) of the best match
    x_curr, y_curr = max_loc

    # Calculate Shift
    shift_x = x_curr - x_base
    shift_y = y_curr - y_base
    
    # Confidence score (0 to 1)
    confidence = max_val 

    return shift_x, shift_y, confidence

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Check camera alignment shifts.")
    
    # Positional argument: The folder you want to test (Current Day)
    parser.add_argument("curr_dir", type=str, help="Path to the CURRENT day's video folder")
    
    # Optional arguments: Override defaults if needed
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Path to BASELINE folder")
    parser.add_argument("--rois", type=str, default=DEFAULT_ROIS_PATH, help="Path to ROI json file")
    
    args = parser.parse_args()

    # 2. Load Resources
    camera_rois = load_rois(args.rois)
    if not camera_rois:
        sys.exit(1)

    print(f"{'Camera':<10} | {'Shift X':<10} | {'Shift Y':<10} | {'Conf':<6}")
    print("-" * 45)

    # 3. Process each camera
    for camera in CAMERAS_LIST:
        # Check if camera exists in our ROI file
        if camera not in camera_rois:
            print(f"{camera:<10} | {'ROI NOT FOUND':<24} |")
            continue

        # --- Load Baseline ---
        video_path_base = os.path.join(args.base_dir, f'{camera}-orig.mp4')
        frame_base = get_single_frame(video_path_base)
        
        if frame_base is None: continue # Skip if base video missing

        gray_base = cv2.cvtColor(frame_base, cv2.COLOR_BGR2GRAY)

        # Prepare Template
        x_base, y_base, w, h = camera_rois[camera]
        template = gray_base[y_base:y_base+h, x_base:x_base+w]

        # --- Load Current ---
        video_path_curr = os.path.join(args.curr_dir, f'{camera}-orig.mp4')
        frame_curr = get_single_frame(video_path_curr)

        if frame_curr is None: continue # Skip if current video missing

        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

        # --- Compare ---
        shift_x, shift_y, conf = quantify_camera_shift(gray_curr, template, x_base, y_base)

        # Format output
        sx_str = f"{shift_x:+d} px"
        sy_str = f"{shift_y:+d} px"
        
        # Add a warning flag if confidence is low
        conf_str = f"{conf:.2f}"
        if conf < 0.8:
            conf_str += " (!)"

        print(f"{camera:<10} | {sx_str:<10} | {sy_str:<10} | {conf_str:<6}")

if __name__ == "__main__":
    main()