import os
from pathlib import Path
from glob import glob
import toml
import json
import numpy as np
import pandas as pd
import argparse


# ------------ Set up parameters and directories -------------

# Define the hand plane
HAND_PLANE_POINTS = ['Wrist_U', 'Palm', 'Wrist_R']
AP_CONFIG_NAME = "config.toml"
# -----------------------------------------------


# ------------ Functions -------------
def get_hand_plane_points(pose_3d_dir, file_name):
    '''
    Arguments:

    pose_3d_dir: Directory containing 3D CSV files.
    file_name: 3D CSV file name.

    Returns:

    hand_plane_coordinates: an array (num_frames, num_keypoints, num_coordinates(x,y,z)) .
    '''

    # Load 3D data
    pose_3d_df = pd.read_csv(os.path.join(pose_3d_dir, file_name))
    # Get palm and wrist points
    hand_plane_coordinates = pose_3d_df[[f"{hp}_x" for hp in HAND_PLANE_POINTS] + 
                                [f"{hp}_y" for hp in HAND_PLANE_POINTS] + 
                                [f"{hp}_z" for hp in HAND_PLANE_POINTS]].values.reshape(-1, 3, 3)
    hand_plane_coordinates = hand_plane_coordinates.transpose(0, 2, 1)

    return np.array(hand_plane_coordinates)

def compute_plane_normals_per_frame(plane_points):
    '''
    Arguments:
    plane_points: Array of shape (num_frames, num_keypoints, num_coordinates(x,y,z)).
    Returns:
    normals: Array of shape (num_frames, 3), normal vectors for each frame.
    '''
    normals = []
    for i in range(plane_points.shape[0]): # Iterate through frames
        vec1 = plane_points[i, 1, :] - plane_points[i, 0, :]
        vec2 = plane_points[i, 2, :] - plane_points[i, 0, :]
        normal = np.cross(vec1, vec2)
        normal /= np.linalg.norm(normal)
        normals.append(normal)

    return np.array(normals)

def process_trial(pose_3d_dir, file_name):

    plane_points = get_hand_plane_points(pose_3d_dir, file_name)

    normals = compute_plane_normals_per_frame(plane_points=plane_points)

    return normals