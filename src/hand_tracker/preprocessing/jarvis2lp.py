import os
import glob
import shutil
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from hand_tracker.preprocessing.convert_videos import get_videos  
from hand_tracker.utils.file_io import contains_subdirectory

# Data preparation for training Lightning Pose models
# 1. Convert Jarvis labeled data to Lightning Pose labeled data 
# 2. Format videos
# 3. Get context frames

CAMERA_VIEWS = ['camTo', 'camTL', 'camTR', 'camBL', 'camBR']

def format_frame_name(raw_name: str) -> str:
    """Standardizes frame naming: 'frame_123.jpg' -> 'frame000123.png'"""
    root = Path(raw_name).stem
    parts = root.split('_')
    frame_num = int(parts[-1])
    return f"frame{frame_num:06d}.png"

def J2LP_csv(csv_file, t, c):
    """Converts Jarvis CSV structure to Lightning Pose MultiIndex DataFrame."""
    # 1. Read the first 4 rows which contain the header info
    df_header = pd.read_csv(csv_file, on_bad_lines='skip', nrows=4, header=None, index_col=0)
    # 2. Read the data rows starting from row 5
    df_data = pd.read_csv(csv_file, skiprows=4, header=None, index_col=0)
    
    # 3. Filter out 'state' columns 
    # Usually, row index 3 in Jarvis contains 'x', 'y', 'state'
    valid_cols = [i for i, val in enumerate(df_header.iloc[3]) if val != 'state']
    df_header = df_header.iloc[:, valid_cols]
    df_data = df_data.iloc[:, valid_cols]
    
    # 4. Standardize the Index (Image Paths)
    new_index = []
    for img_name in df_data.index:
        fname = format_frame_name(img_name)
        # Using original trial name 't' as requested
        new_index.append(f"labeled-data/{t}/{c}/{fname}")
    df_data.index = new_index
    
    # 5. Construct MultiIndex
    # We remove the row named 'entities' from the header to match LP/DLC requirements
    # LP expects 3 levels: scorer, bodyparts, coords (x, y)
    header_rows_to_keep = [row for row in df_header.index if row != 'entities']
    filtered_header = df_header.loc[header_rows_to_keep]
    
    multi_columns = pd.MultiIndex.from_arrays(filtered_header.values)
    
    # 6. Return the formatted DataFrame
    return pd.DataFrame(df_data.values, index=df_data.index, columns=multi_columns)

def get_labeled_frames(jarvis_dir: str, lp_dir: str):

    print(f"Get labeled frames: {jarvis_dir}")

    jarvis_path = Path(jarvis_dir)
    lp_path = Path(lp_dir)
    trials = sorted([d.name for d in jarvis_path.iterdir() if d.is_dir()])
    
    for t in trials:
        for c in CAMERA_VIEWS:
            src_dir = jarvis_path / t / c
            if not src_dir.exists(): continue
            
            # Creating folder with original trial name 't'
            dest_dir = lp_path / "labeled-data" / t / c
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in src_dir.glob("*.jpg"):
                with Image.open(img_path) as im:
                    im.save(dest_dir / format_frame_name(img_path.name))

def get_context_frames(lp_dir, context_range):
    """
    Extracts neighboring frames from videos to provide temporal context.
    Assumes video filenames match the original trial folder names (e.g., Trialname_Viewname.mp4).
    """
    print(f"Get context frames: {lp_dir}")

    lp_path = Path(lp_dir)
    video_dir = lp_path / "videos"
    
    for video_file in video_dir.glob("*.mp4"):
        cap = cv2.VideoCapture(str(video_file))

        parts = video_file.stem.split('_')
        view_name = parts[-1]
        trial_name = "_".join(parts[:-1])
        
        frame_folder = lp_path / "labeled-data" / trial_name / view_name
        if not frame_folder.exists():
            continue

        # Extract frame numbers from 'frame000123.png'
        # We find all files starting with 'frame' and ending with '.png'
        existing_frames = []
        for f in frame_folder.glob("frame*.png"):
            try:
                # Extracts the digits between 'frame' and '.png'
                num = int(f.stem.replace('frame', ''))
                existing_frames.append(num)
            except ValueError:
                continue
        
        for fr_idx in existing_frames:
            for offset in range(context_range[0], context_range[1] + 1):
                target_frame = fr_idx + offset
                if target_frame < 0: continue
                
                out_path = frame_folder / f"frame{target_frame:06d}.png"
                
                # Only extract if the context frame doesn't already exist
                if not out_path.exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(str(out_path), frame)
        cap.release()

def J2LP_sigleview(jarvis_dir: str, lp_dir: str):
    jarvis_path = Path(jarvis_dir)
    trials = sorted([d.name for d in jarvis_path.iterdir() if d.is_dir()])
    
    all_dfs = []
    for t in trials:
        for c in CAMERA_VIEWS:
            csv_path = jarvis_path / t / c / "annotations.csv"
            if csv_path.exists():
                df = J2LP_csv(str(csv_path), t, c)
                all_dfs.append(df)
    
    if all_dfs:
        df_final = pd.concat(all_dfs)
        Path(lp_dir).mkdir(parents=True, exist_ok=True)
        df_final.to_csv(Path(lp_dir) / "CollectedData.csv")


def J2LP_sigview_multisession(jarvis_annotations_list: list, lp_dir: str):
    """
    Aggregates annotations from multiple Jarvis session directories into a 
    single Lightning Pose CollectedData.csv.
    """
    all_dfs = []
    lp_path = Path(lp_dir)
    
    for jarvis_dir in jarvis_annotations_list:
        jarvis_path = Path(jarvis_dir)
        
        # Identify trial directories
        trials = sorted([d.name for d in jarvis_path.iterdir() if d.is_dir() and contains_subdirectory(d)])
        
        for t in trials:            
            for c in CAMERA_VIEWS:
                csv_path = jarvis_path / t / c / "annotations.csv"
                
                if csv_path.exists():
                    # J2LP_csv now returns a MultiIndex DataFrame directly
                    df_lp = J2LP_csv(str(csv_path), t, c)
                    all_dfs.append(df_lp)
                else:
                    print(f"Warning: {csv_path} not found. Skipping.")

    if all_dfs:
        # Concatenate all sessions and trials into one master dataframe
        df_all = pd.concat(all_dfs)
        
        # Ensure output directory exists and save
        lp_path.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(lp_path / "CollectedData.csv")
        print(f"Successfully aggregated {len(all_dfs)} sets of annotations to {lp_path}")
    else:
        print("No annotations were found to aggregate.")


def J2LP_multiview(jarvis_dir: str, lp_dir: str):
    """
    Converts Jarvis data to Lightning Pose format, creating separate 
    CollectedData files for each camera view across all trials.
    """
    jarvis_path = Path(jarvis_dir)
    lp_path = Path(lp_dir)
    lp_path.mkdir(parents=True, exist_ok=True)

    # Find trial folders
    trials = sorted([d.name for d in jarvis_path.iterdir() if d.is_dir() and contains_subdirectory(d)])
    
    if not trials:
        print("No trial directories found.")
        return

    for c in CAMERA_VIEWS:
        dfs_for_camera = []
        for t in trials:
            csv_path = jarvis_path / t / c / "annotations.csv"
            
            if csv_path.exists():
                # Directly process the CSV into a MultiIndex DataFrame
                df_lp = J2LP_csv(str(csv_path), t, c)
                dfs_for_camera.append(df_lp)
            else:
                print(f"Warning: Missing annotations for trial {t}, camera {c}")

        if dfs_for_camera:
            # Concatenate all trials for this specific camera
            df_all_camera = pd.concat(dfs_for_camera)
            
            # Save as camera-specific CSV (e.g., CollectedData_cam0.csv)
            output_filename = f"CollectedData_{c}.csv"
            df_all_camera.to_csv(lp_path / output_filename)
            print(f"Created {output_filename} with {len(dfs_for_camera)} trials.")


def J2LP_multiview_multisession(jarvis_annotations_list: list, lp_dir: str):
    lp_path = Path(lp_dir)
    lp_path.mkdir(parents=True, exist_ok=True)

    for c in CAMERA_VIEWS:
        dfs_for_camera = []
        for jarvis_dir in jarvis_annotations_list:
            jarvis_path = Path(jarvis_dir)
            trials = sorted([d.name for d in jarvis_path.iterdir() if d.is_dir() and contains_subdirectory(d)])

            for t in trials:
                csv_path = jarvis_path / t / c / "annotations.csv"
                if csv_path.exists():
                    df_lp = J2LP_csv(str(csv_path), t, c)
                    dfs_for_camera.append(df_lp)
        
        if dfs_for_camera:
            df_all_camera = pd.concat(dfs_for_camera)
            output_filename = f"CollectedData_{c}.csv"
            df_all_camera.to_csv(lp_path / output_filename)
            print(f"Created {output_filename} for camera {c}")

def create_calibration_index_file(lp_dir: str, camera_view: str = "camTo"):
    lp_path = Path(lp_dir)
    # 1. Load your existing labels to get the CORRECT order of images
    # We use camTo as the reference for the image list
    label_csv = lp_path / f"CollectedData_{camera_view}.csv"
    df_labels = pd.read_csv(label_csv, header=[0, 1, 2], index_col=0)
    
    labeled_img_paths = []
    cal_paths = []

    for full_path in df_labels.index:
        # Transform: labeled-data/Trial_01/camTo/frame000001.png 
        # To: labeled-data/Trial_01/frame000001.png
        p = Path(full_path)
        trial_name = p.parent.parent.name
        frame_name = p.name
        
        # View-agnostic path
        agnostic_path = f"labeled-data/{trial_name}/{frame_name}"
        labeled_img_paths.append(agnostic_path)
        
        # Calibration file path (e.g., calibrations/Trial_01.toml)
        cal_paths.append(f"calibrations/{trial_name}.toml")

    # 2. Build the dataframe to match the required format
    df_cal = pd.DataFrame(cal_paths, index=labeled_img_paths, columns=['file'])
    
    # Save to lp_dir/calibrations.csv
    df_cal.to_csv(lp_path / "calibrations.csv")

def create_calibration_files(jarvis_annotations_list: list, lp_dir: str):
    lp_path = Path(lp_dir)
    # Ensure the calibrations subfolder exists in the LP project
    lp_cal_dir = lp_path / "calibrations"
    lp_cal_dir.mkdir(parents=True, exist_ok=True)

    for jarvis_dir in jarvis_annotations_list:
        jarvis_path = Path(jarvis_dir)
        trials = sorted([d.name for d in jarvis_path.iterdir() if d.is_dir() and contains_subdirectory(d)])
        
        # Source calibration file (assuming one per session/jarvis_dir)
        anipose_cal_src = jarvis_path / "calibration_anipose" / "calibration.toml"
        
        for t in trials:
            # 1. Handle Calibration File
            # We save the calibration file in the LP project
            cal_filename = f"{t}.toml"
            cal_dest = lp_cal_dir / cal_filename
            if anipose_cal_src.exists():
                shutil.copy(anipose_cal_src, cal_dest)

def main(jarvis_annotations_dir = None, lp_dir = None, view_mode = 'single', calibration_mode = 'single', context_mode = True):

    jarvis_annotations_dirs = glob.glob(os.path.join(jarvis_annotations_dir, 'annotations_2508*'))  # list of Jarvis labeled datasets

    ## 1. onvert Jarvis labeled data to Lightning Pose labeled data
    if view_mode == 'singleview':
        # --------- Single-view format ---------
        J2LP_sigview_multisession(jarvis_annotations_dirs, lp_dir)

        # Copy labeled frames to the LP project folder
        for jarvis_dir in jarvis_annotations_dirs:
            get_labeled_frames(jarvis_dir, lp_dir)

        # Check if frame paths are correct
        csv_file = os.path.join(lp_dir, "CollectedData.csv")
        df_all = pd.read_csv(csv_file, header = [0,1,2], index_col=0)
        for im in df_all.index:
            assert os.path.exists(os.path.join(lp_dir, im))

    elif view_mode == 'multiview':
        # --------- Multi-view format ---------
        # Convert Jarvis labeled data to Lightning Pose labeled data
        J2LP_multiview_multisession(jarvis_annotations_dirs, lp_dir)

        # Copy labeled frames to the LP project folder
        for jarvis_dir in jarvis_annotations_dirs:
            get_labeled_frames(jarvis_dir, lp_dir)

        # Check if frame paths are correct
        camera_csvs = [filename for filename in os.listdir(lp_dir) if filename.endswith('.csv')]
        for c in camera_csvs:
            csv_file = os.path.join(lp_dir, c)
            df_all = pd.read_csv(csv_file, header = [0,1,2], index_col=0)
            for im in df_all.index:
                assert os.path.exists(os.path.join(lp_dir, im))

        if calibration_mode == 'multiple':
            # Create calibration files (optional if multiple calibrations)
            create_calibration_files(jarvis_annotations_dirs, lp_dir)
            create_calibration_index_file(lp_dir=lp_dir)
        else : # Project-wise single calibration
            anipose_cal_src = Path(jarvis_annotations_dirs[0]) / "calibration_anipose" / "calibration.toml"
            cal_dest = Path(lp_dir) / "calibration.toml"
            shutil.copy(anipose_cal_src, cal_dest)

    ## 2. Format, rename, organize videos
    src_vid_dir = r'/media/yiting/NewVolume/Data/Videos'
    for jarvis_dir in jarvis_annotations_dirs:
        get_videos(jarvis_dir, lp_dir, src_vid_dir)
    
    # ## 3. Get context frames (optional)
    # As of March 2026, multi-view Lightning Pose does not yet support context frames or unsupervised losses.
    if context_mode:
        context_range = [-2, 2]
        get_context_frames(lp_dir, context_range)

if __name__ == "__main__":

    jarvis_annotations_dir = r'/home/yiting/Documents/Jarvis/datasets/annotations' # path to the Jarvis labeled datasets
    lp_dir = r'/home/yiting/Documents/GitHub/lightning-pose/data/test_multiview_singlecal_2508' # path to the lp project

    main(
        jarvis_annotations_dir=jarvis_annotations_dir,
        lp_dir=lp_dir,
        view_mode='multiview',
        calibration_mode='single',
        context_mode=False
        )
