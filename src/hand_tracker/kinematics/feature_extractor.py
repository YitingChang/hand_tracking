# Feature extraction
# 1. Coordinates
# 2. Angles
# 3. Hand orientation
# 4. Hand and object distance and angle

import os
from pathlib import Path
import toml
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
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

OBJECT_KEYPOINTS = ["Dot_t1", "Dot_t2", "Dot_t3", "Dot_b1", "Dot_b2", "Dot_b3", 
                    "Dot_l1", "Dot_l2", "Dot_l3", "Dot_r1", "Dot_r2", "Dot_r3"]

OBJECT_REF_KEYPOINT = ["Dot_t2"] # Reference point on the object for hand-object distance

AP_CONFIG_NAME = "config.toml"

def compute_relational_features(pose_3d_df):
    """
    Main wrapper to extract both types of relative features.
    """
    # 1. Hand-Object Distances (Vector of 23 features per trial)
    df_hand_obj = compute_hand_object_distance(pose_3d_df)
    
    # 2. Hand Pairwise Distances (Vector of 253 features per trial)
    df_hand_pair = compute_hand_pairwise_dist(pose_3d_df)
    
    # Combine them into a single feature matrix for the RDM
    df_relational = pd.concat([df_hand_obj, df_hand_pair], axis=1)
    return df_relational

def compute_hand_object_distance(pose_3d_df):
    """
    Computes distance between hand keypoints and the object reference point.
    """
    # Vectorize for speed instead of nested loops
    distances = []
    # Reference object point
    ref_x, ref_y, ref_z = f"{OBJECT_REF_KEYPOINT[0]}_x", f"{OBJECT_REF_KEYPOINT[0]}_y", f"{OBJECT_REF_KEYPOINT[0]}_z"
    
    for _, trial_row in pose_3d_df.iterrows():
        opt = np.array([trial_row[ref_x], trial_row[ref_y], trial_row[ref_z]])
        trial_dists = []
        for hk in HAND_KEYPOINTS:
            hpt = np.array([trial_row[f"{hk}_x"], trial_row[f"{hk}_y"], trial_row[f"{hk}_z"]])
            trial_dists.append(np.linalg.norm(hpt - opt))
        distances.append(trial_dists)

    return pd.DataFrame(distances, columns=[f"dist_obj_{hk}" for hk in HAND_KEYPOINTS])

def compute_hand_pairwise_dist(pose_3d_df):
    """
    Computes pairwise distance between all 23 hand keypoints.
    This describes the 'conformation' or 'shape' of the hand regardless of position.
    """
    pairwise_feats = []
    for _, trial_row in pose_3d_df.iterrows():
        # Extract (23, 3) matrix of hand keypoints
        hand_pts = []
        for hk in HAND_KEYPOINTS:
            hand_pts.append([trial_row[f"{hk}_x"], trial_row[f"{hk}_y"], trial_row[f"{hk}_z"]])
        hand_pts = np.array(hand_pts)
        
        # pdist computes the upper triangle of the distance matrix
        # For 23 points, this is (23*22)/2 = 253 distances
        dists = pdist(hand_pts, metric='euclidean')
        pairwise_feats.append(dists)
        
    # Create column names: e.g., 'pair_Wrist_R_Palm'
    from itertools import combinations
    col_names = [f"pair_{a}_{b}" for a, b in combinations(HAND_KEYPOINTS, 2)]
    
    return pd.DataFrame(pairwise_feats, columns=col_names)

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

    angle_dir = ap_dir / ap_config['pipeline']['angles']
    os.makedirs(angle_dir, exist_ok = True)

    feature_dir = analysis_dir / session_name / "features"
    os.makedirs(feature_dir, exist_ok = True)

    for pose_3d_file in pose_3d_files:
        # 0. Load 3D coordinates data
        pose_3d_path = pose_3d_dir / pose_3d_file
        pose_3d_df = pd.read_csv(pose_3d_path)
        trial_name = get_trialname(pose_3d_file)

        # 1. Calculate Keypoint Distances: pairwise distances between hand keypoints and hand-object distances
        distance_df = compute_relational_features(pose_3d_df)

        # 2. Compute Angles
        angle_out_fname = os.path.join(angle_dir, trial_name + '_angles.csv')
        angle_df = compute_angles(ap_config, pose_3d_path, angle_out_fname)

        # 3. Compute Hand Orientations
        normals, hand_obj_angles = orientation_process(pose_3d_dir, pose_3d_file)

        # 5. Combine features (coordinates, angles, hand orientations)  
        feature_df = pose_3d_df.copy()
        feature_df = pd.concat([distance_df, angle_df.iloc[:, :-1]], axis=1)
        feature_df["normal_x"] = normals[:,0]
        feature_df["normal_y"] = normals[:,1]
        feature_df["normal_z"] = normals[:,2]
        feature_df["hand_obj_angle"] = hand_obj_angles

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
