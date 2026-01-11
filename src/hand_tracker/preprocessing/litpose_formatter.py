import os
import shutil
import argparse
from pathlib import Path

# Constants
CAMERA_VIEWS = ["To", "TL", "TR", "BL", "BR"]

def prep_litpose(session_name, data_dir, analysis_dir):
    """
    Reorganizes raw video files into the Lightning Pose directory structure.
    """
    print(f"Formatting videos for Lightning-Pose inference: {session_name}")
    
    # Convert strings to Path objects
    data_dir = Path(data_dir)
    analysis_dir = Path(analysis_dir)

    # Define source and destination
    # Assuming structure: data_dir / session_name / cameras / trial_folders
    session_input_dir = data_dir / session_name / "cameras"
    
    # Destination: analysis_dir / session_name / litpose / new_videos
    lp_video_dir = analysis_dir / session_name / "litpose" / "new_videos"
    lp_video_dir.mkdir(parents=True, exist_ok=True)

    if not session_input_dir.exists():
        print(f"Error: Input directory not found: {session_input_dir}")
        return

    trials = sorted([t for t in os.listdir(session_input_dir) if not t.startswith('.')])

    for t in trials:
        trialname_parts = t.split('_')
        
        # Safety check for folder naming convention
        if len(trialname_parts) < 2:
            print(f"Skipping malformed folder name: {t}")
            continue

        for c in CAMERA_VIEWS:
            # Source: .../cameras/trial_folder/camTo.mp4
            raw_vid_name = f"cam{c}.mp4"
            raw_vid_file = session_input_dir / t / raw_vid_name

            if raw_vid_file.exists():
                new_vid_name = f"{trialname_parts[0]}_{trialname_parts[1]}_cam{c}.mp4"
                lp_vid_file = lp_video_dir / new_vid_name
                
                print(f"Copying {raw_vid_name} -> {new_vid_name}")
                shutil.copyfile(raw_vid_file, lp_vid_file)
            else:
                print(f"Warning: Missing file {raw_vid_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True, help="Name of the session (e.g., 2025-12-09)")
    parser.add_argument("--data_dir", type=str, required=True, help="Root path to raw data")
    parser.add_argument("--analysis_dir", type=str, required=True, help="Root path to analysis output")
    
    args = parser.parse_args()
    
    prep_litpose(args.session, args.data_dir, args.analysis_dir)