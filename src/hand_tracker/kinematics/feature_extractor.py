# Feature extraction
# 1. Angles
# 2. Hand orientation
import os
from pathlib import Path
import toml
import argparse
import pandas as pd

from hand_tracker.utils.file_io import get_trialname
from hand_tracker.anipose_yt.compute_angles import compute_angles
from hand_tracker.kinematics.hand_orientation import process_trial as orientation_process


# Constants
CAMERA_VIEWS = ["To", "TL", "TR", "BL", "BR"]

HAND_KEYPOINTS = ["Small_Tip", "Small_DIP", "Small_PIP", "Small_MCP", 
                  "Ring_Tip", "Ring_DIP", "Ring_PIP", "Ring_MCP", 
                  "Middle_Tip", "Middle_DIP", "Middle_PIP", "Middle_MCP", 
                  "Index_Tip", "Index_DIP", "Index_PIP", "Index_MCP", 
                  "Thumb_Tip", "Thumb_IP", "Thumb_MCP" ,"Thumb_CMC", 
                  "Palm", "Wrist_U","Wrist_R"]

AP_CONFIG_NAME = "config.toml"


def main(session_name, analysis_dir):

    analysis_dir = Path(analysis_dir)
    ap_dir = analysis_dir / session_name / "anipose"

    # 0. Set up paths and load configs
    ap_config_path = ap_dir / AP_CONFIG_NAME
    ap_config = toml.load(ap_config_path)

    if ap_config['filter3d']['enabled']:
        pose_3d_dir = ap_dir / "pose_3d_filter"
    else:
        pose_3d_dir = ap_dir / "pose_3d"
    pose_3d_files = sorted(os.listdir(pose_3d_dir))

    feature_dir = analysis_dir / session_name / "features"
    os.makedirs(feature_dir, exist_ok = True)

    for pose_3d_file in pose_3d_files:
        # 0. Load 3D coordinates data
        pose_3d_path = pose_3d_dir / pose_3d_file
        pose_3d_df = pd.read_csv(pose_3d_path)
        trial_name = get_trialname(pose_3d_file)

        # 1. Get Hand Coordinates
        # Get coordinates of hand keypoints 
        hand_df = pose_3d_df[[f"{hp}_x" for hp in HAND_KEYPOINTS] + 
                                [f"{hp}_y" for hp in HAND_KEYPOINTS] + 
                                [f"{hp}_z" for hp in HAND_KEYPOINTS]]

        # 2. Compute Angles
        angle_out_fname = os.path.join(ap_dir, ap_config['pipeline']['angles'], trial_name + '_angles.csv')
        angle_df = compute_angles(ap_config, pose_3d_path, angle_out_fname)

        # 3. Compute Hand Orientations
        normals = orientation_process(pose_3d_dir, pose_3d_file)

        # 5. Combine features (coordinates, angles, hand orientations)  
        feature_df = pose_3d_df.copy()
        feature_df = pd.concat([hand_df, angle_df.iloc[:, :-1]], axis=1)
        feature_df["normal_x"] = normals[:,0]
        feature_df["normal_y"] = normals[:,1]
        feature_df["normal_z"] = normals[:,2]

        # 6. Save 
        out_fname = os.path.join(feature_dir, trial_name + '_features.csv')
        feature_df.to_csv(out_fname, index=False)
        print(out_fname)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True, help="Name of the session (e.g., 2025-12-09)")
    parser.add_argument("--analysis_dir", type=str, required=True, help="Root path to analysis output")
    
    args = parser.parse_args()
    
    main(args.session, args.analysis_dir)
