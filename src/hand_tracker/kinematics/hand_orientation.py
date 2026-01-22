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
OBJECT_PLANE_POINTS = ["Dot_t2", "Dot_t3", "Dot_b2"]
AP_CONFIG_NAME = "config.toml"
# -----------------------------------------------


# ------------ Functions -------------
def get_plane_points(pose_3d_dir, file_name, plane_keypoints):
    '''
    Arguments:

    pose_3d_dir: Directory containing 3D CSV files.
    file_name: 3D CSV file name.
    plane_keypoints = Three keypoints to define a plane 

    Returns:

    plane_coordinates: an array (num_frames, num_keypoints , num_coordinates(x,y,z)) .
    '''

    # Load 3D data
    pose_3d_df = pd.read_csv(os.path.join(pose_3d_dir, file_name))
    # Get plane keypoints
    plane_coordinates = pose_3d_df[[f"{hp}_x" for hp in plane_keypoints] + 
                                [f"{hp}_y" for hp in plane_keypoints] + 
                                [f"{hp}_z" for hp in plane_keypoints]].values.reshape(-1, 3, 3)
    plane_coordinates = plane_coordinates.transpose(0, 2, 1)

    return np.array(plane_coordinates)

def compute_plane_normals_per_frame(plane_coordinates):
    '''
    Arguments:
    plane_coordinates: Array of shape (num_frames, num_keypoints, num_coordinates(x,y,z)).
    Returns:
    normals: Array of shape (num_frames, 3), normal vectors for each frame.
    '''
    normals = []
    for i in range(plane_coordinates.shape[0]): # Iterate through frames
        vec1 = plane_coordinates[i, 1, :] - plane_coordinates[i, 0, :]
        vec2 = plane_coordinates[i, 2, :] - plane_coordinates[i, 0, :]
        normal = np.cross(vec1, vec2)
        normal /= np.linalg.norm(normal)
        normals.append(normal)

    return np.array(normals)

def compute_plane_angles(normals_a, normals_b, degrees=True):
    """
    Computes angles between two lists of normal vectors.
    
    Args:
        normals_a: (N, 3) array of normal vectors for plane set A
        normals_b: (N, 3) array of normal vectors for plane set B
        degrees: Return angle in degrees if True, radians otherwise
    
    Returns:
        angles: (N,) array of angles
    """
    # 1. Row-wise Dot Product
    # Multiply element-wise and sum along axis 1
    dot_products = np.sum(normals_a * normals_b, axis=1)
    
    # 2. Row-wise Magnitudes (Norms)
    norms_a = np.linalg.norm(normals_a, axis=1)
    norms_b = np.linalg.norm(normals_b, axis=1)
    
    # 3. Compute Cosine of the Angle
    # Divide dot product by product of magnitudes
    # We use np.clip to handle floating point errors (e.g., 1.00000002)
    cos_angles = dot_products / (norms_a * norms_b)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    
    # 4. Compute Angle (Arccos)
    angles = np.arccos(cos_angles)
    
    # OPTIONAL: If you strictly want the ACUTE angle (0-90 degrees)
    # between the planes regardless of vector orientation:
    # angles = np.minimum(angles, np.pi - angles)
    
    # 5. Convert to Degrees
    if degrees:
        angles = np.degrees(angles)
        
    return angles

def process_trial(pose_3d_dir, file_name):

    # Compute normals of hand plane 
    hand_plane_coordinates = get_plane_points(pose_3d_dir, file_name, HAND_PLANE_POINTS)

    hand_normals = compute_plane_normals_per_frame(plane_coordinates=hand_plane_coordinates)

    # Compute normals of object plane
    object_plane_coordinates = get_plane_points(pose_3d_dir, file_name, OBJECT_PLANE_POINTS)

    object_normals = compute_plane_normals_per_frame(plane_coordinates=object_plane_coordinates)

    # Compute the angles between hand and object planes
    hand_obj_angles = compute_plane_angles(hand_normals, object_normals, degrees=True)

    return hand_normals, hand_obj_angles