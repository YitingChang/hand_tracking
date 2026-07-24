# Get the middle-of-hold-window frame from lighting pose videos

import cv2
import os   
from pathlib import Path
import json
import numpy as np
import pandas as pd
from glob import glob
from hand_tracker.utils.file_io import get_trialname, find_matching_log
from hand_tracker.utils.analysis_window import load_window_lookup


# ==========================================
# 0. GLOBAL PATHS & CONFIGURATIONS
# ==========================================
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
STL_ROOT = Path("/media/yiting/NewVolume/Data/Shapes/shapes_stl")
CONFIG_JSON_PATH = Path("/home/yiting/Documents/GitHub/hand_tracking/configs/obj_coordinates.json")

CAMERA_NAMES = ["camTo", "camTL", "camTR", "camBR", "camBL"]  # List of camera identifiers

# ==========================================
# 1. IO AND DATA PROCESSING MODULE
# ==========================================
def load_trial_metadata(log_fname):
    """Extracts object mapping ID and trial stimulus orientation from log files."""
    with open(log_fname, 'r') as file:
        log_data = json.load(file)
    shape_id = log_data.get("shape_id", "unknown_0")
    obj_id = shape_id.split("_")[0]
    orientation = shape_id.split("_")[-1]
    return obj_id, orientation, shape_id

# ==========================================
# 2. PIPELINE CONTROLLER EXECUTION
# ==========================================
def process_single_trial(session_name, trial_name, log_fname, frame_number):
    """Runs the execution cycle for an isolated context frame scenario."""
    print(f"-> Commencing Analysis Pipeline: Trial '{trial_name}' | Frame [{frame_number}]")
    
    # 1. Load Configurations and videos
    obj_id, orientation, shape_id = load_trial_metadata(log_fname)
    
    recon_dir = ANALYSIS_ROOT / session_name / 'reconstructions' / trial_name
    recon_dir.mkdir(parents=True, exist_ok=True)

    for cam in CAMERA_NAMES:
        # video_path = ANALYSIS_ROOT / session_name / "litpose" / "video_preds" / f"{trial_name}_{cam}_labeled.mp4"
        video_path = ANALYSIS_ROOT / session_name / "litpose" / "new_videos" / f"{trial_name}_{cam}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_number} from video: {video_path}")
        
        # Save the extracted frame as an image

        # frame_output_path = recon_dir / f"{trial_name}_{cam}_labeled_f{frame_number}.png"
        frame_output_path = recon_dir / f"{trial_name}_{cam}_f{frame_number}.png"
        cv2.imwrite(str(frame_output_path), frame)
        cap.release()


def batch_process_session(session_name, trial_names, log_fnames, window_lookup):
    print(f"=== Initiating Batch Processing Session: {session_name} ===")
    total_trials = len(trial_names)
    skipped_no_window = 0
    
    for idx, [trial_name, log_fname] in enumerate(zip(trial_names, log_fnames)):
        print(f"Progress: [{idx + 1}/{total_trials}]")
        
        try:
            with open(log_fname, 'r') as file:
                log_data = json.load(file)
                has_halted_motion = log_data.get("has_halted_motion", False)
        except Exception as e:
            print(f"❌ Error occurred while reading log file: {log_fname}. Details: {e}\n")
            continue

        if not has_halted_motion:
            print(f"❌ Skipping Trial '{trial_name}' due to no detected grabbing motion.\n")
            continue

        window = window_lookup.get(trial_name)
        if window is None or pd.isna(window[0]) or pd.isna(window[1]):
            print(f"❌ Skipping Trial '{trial_name}' due to no hold window on record.\n")
            skipped_no_window += 1
            continue

        start_frame, end_frame = int(window[0]), int(window[1])
        if start_frame > end_frame:
            print(f"❌ Skipping Trial '{trial_name}' due to invalid hold window [{start_frame}, {end_frame}].\n")
            skipped_no_window += 1
            continue

        middle_frame = (start_frame + end_frame) // 2

        try:
            process_single_trial(session_name, trial_name, log_fname, middle_frame)
        except Exception as e:
            print(f"❌ Error occurred while processing Trial: '{trial_name}'. Details: {e}\n")
            continue
            
    if skipped_no_window:
        print(f"Skipped {skipped_no_window} trials with no hold window on record.")
    print("=== Batch Processing Sequence Completed ===")

# ==========================================
# 7. MAIN ROUTINE ENTRY POINT
# ==========================================
if __name__ == "__main__":
    session_names = ["2025-12-09"]
    # session_names = ["2025-08-19", "2025-08-22", "2025-11-20",
    #                   "2025-12-08", "2025-12-09", "2025-12-18"]
    for session_name in session_names:
        feature_dir = os.path.join(ANALYSIS_ROOT, session_name, "features")
        log_dir = os.path.join(RAW_DATA_ROOT, session_name, "trial_logs")

        window_lookup = load_window_lookup(session_name)
        if window_lookup is None:
            print(f"Warning: no min_holding_window.csv found for {session_name}, skipping session.")
            continue

        feature_fnames = sorted(glob(os.path.join(feature_dir, "*.csv")))
        log_fnames = find_matching_log(feature_fnames, log_dir)
        trial_names = [get_trialname(f) for f in feature_fnames]

        # Execute batch pipeline
        batch_process_session(session_name, trial_names, log_fnames, window_lookup)