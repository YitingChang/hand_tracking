import os
import subprocess
import shutil
from pathlib import Path
from hand_tracker.utils.file_io import contains_subdirectory

def check_codec_format(input_file: str) -> bool:
    """Run FFprobe command to get video codec and pixel format."""
    # Using list-based subprocess call is safer for paths with spaces
    ffmpeg_cmd = ['ffmpeg', '-i', str(input_file)]
    output = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    # search for correct codec (h264) and pixel format (yuv420p)
    return 'h264' in output.stderr and 'yuv420p' in output.stderr

def reencode_video(input_file: str, output_file: str) -> None:
    """Reencodes video into H.264 coded format."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Standard Lightning Pose compatible re-encoding
    ffmpeg_cmd = [
        'ffmpeg', '-i', str(input_file),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        '-y', str(output_file)
    ]
    subprocess.run(ffmpeg_cmd)

def get_videos(jarvis_dir: str, lp_dir: str, src_vid_dir: str):
    """
    Copies or re-encodes videos from Jarvis source to LP project directory.
    Maintains original trial names and standardizes naming to Trial_01_camName.mp4.
    """
    lp_video_dir = Path(lp_dir) / 'videos'
    lp_video_dir.mkdir(parents=True, exist_ok=True)
    
    jarvis_path = Path(jarvis_dir)
    src_vid_path = Path(src_vid_dir)

    # Find trial folders in JARVIS labeled dataset 
    trials = sorted([d.name for d in jarvis_path.iterdir() if d.is_dir() and contains_subdirectory(d)])

    for t in trials:
        # Assuming src_vid_dir structure: src_vid_dir / SubjectName / 'cameras' / Trial_Name / camera_file
        # We extract 'SubjectName' from the start of the trial string if needed
        # Or if src_vid_dir already points to the camera parent, adjust accordingly.
        subject_name = t.split('_')[0]
        trial_vid_folder = src_vid_path / subject_name / 'cameras' / t
        
        if not trial_vid_folder.exists():
            print(f"Warning: Video folder not found: {trial_vid_folder}")
            continue

        # Iterate through camera files in the source video folder
        for vid_file in trial_vid_folder.iterdir():
            if vid_file.is_dir() or vid_file.suffix not in ['.mp4', '.avi', '.mov', '.mkv']:
                continue
            
            # Identify camera name (e.g., 'camTo' from 'camTo.mp4')
            camera_name = vid_file.stem 
            
            # New name: Trial_01_camTo.mp4 (standardizing extension to .mp4)
            vidname_new = f"{t}_{camera_name}.mp4"
            dest_path = lp_video_dir / vidname_new
            
            print(f"Processing: {t} | {camera_name}")

            # Check video format and reencode if needed
            if check_codec_format(str(vid_file)):
                shutil.copy(str(vid_file), str(dest_path))
            else:
                reencode_video(str(vid_file), str(dest_path))

    print(f"✅ Video processing complete. Videos stored in: {lp_video_dir}")