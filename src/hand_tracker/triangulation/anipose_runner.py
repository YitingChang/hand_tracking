import sys
from pathlib import Path

# --- Dynamic Path Injection ---
# This adds 'src' to sys.path so we can import 'hand_tracker' 
# regardless of which conda env is running this script.
FILE = Path(__file__).resolve()
SRC_ROOT = FILE.parents[2]  # goes up from triangulation -> hand_tracker -> src
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
# ---------------------------------------------------

import os
import shutil
import toml
import argparse

from hand_tracker.utils.file_io import load_litpose_config
from hand_tracker.triangulation.lp2anipose import lp2anipose_session
from hand_tracker.anipose_yt.triangulate import process_trial as triangulate_process
from hand_tracker.anipose_yt.filter_pose import process_trial as filter2d_process
from hand_tracker.anipose_yt.filter_3d import process_trial as filter3d_process

# Constants
CAMERA_VIEWS = ["To", "TL", "TR", "BL", "BR"]
LP_CONFIG_NAME = "config.yaml"
AP_CONFIG_NAME = "config.toml"

def move_lp_preds(source_path, destination_path):

    destination_path.mkdir(parents=True, exist_ok=True)
    for item in source_path.iterdir():
        destination_item = destination_path / item.name
        
        try:
            shutil.move(str(item), str(destination_item))
            print(f"Moved: {item.name}")
        except shutil.Error as e:
            print(f"Error moving {item.name}: {e}")

def main(session_name, analysis_dir):

    analysis_dir = Path(analysis_dir)
    lp_dir = analysis_dir / session_name / "litpose"
    ap_dir = analysis_dir / session_name / "anipose"

    # 0. Load litpose and anipose configs
    lp_config_path = lp_dir / LP_CONFIG_NAME
    lp_config = load_litpose_config(lp_config_path)
    ap_config_path = ap_dir / AP_CONFIG_NAME
    ap_config = toml.load(ap_config_path)

    # 1. Move litpose outputs from the litpose root folder to the analysis session folder

    lp_model_dir = Path(lp_config["eval"]["hydra_paths"][0])
    lp_preds_source_dir = lp_model_dir / "video_preds"
    lp_2d_dir = lp_dir / "video_preds"
    move_lp_preds(lp_preds_source_dir, lp_2d_dir)

    # 2. Convert Lightning pose 2d outputs (.csv) to Anipose inputs (.hdf)
    ap_2d_dir = ap_dir / "pose_2d"
    os.makedirs(ap_2d_dir, exist_ok = True)
    lp2anipose_session(lp_2d_dir, ap_2d_dir, CAMERA_VIEWS)
    trials = sorted(os.listdir(ap_2d_dir))

    # 3. Filter 2D data
    if ap_config['filter']['enabled']:
        for t in trials:
            filter2d_process(ap_config, session_name, t)

    # 4. Triangulation
    for t in trials:
        triangulate_process(ap_config, session_name, t)
    
    # 5. Filter 3D data
    if ap_config['filter3d']['enabled']:
        for t in trials:
            filter3d_process(ap_config, session_name, t)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True, help="Name of the session (e.g., 2025-12-09)")
    parser.add_argument("--analysis_dir", type=str, required=True, help="Root path to analysis output")
    
    args = parser.parse_args()
    
    main(args.session, args.analysis_dir)
