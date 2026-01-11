import os
from glob import glob
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ------------ Set up parameters and directories -------------

# Define the reference plane using top and bottom points
object_points = ['Dot_b2', 'Dot_t2', 'Dot_t3']
hand_points = ['Wrist_U', 'Palm', 'Wrist_R']

frame_rate = 100  # frames per second
time_window = [3, 3.5]

data_dir = r"/media/yiting/NewVolume/Data/Videos"
analysis_dir = r"/media/yiting/NewVolume/Analysis"
# -----------------------------------------------


# ------------ Functions -------------

# Function to extract object and hand plane points from 3D filtered data
def get_object_hand_results(main_dir, file_names):
    '''
    Arguments:

    main_dir: Directory containing 3D filtered CSV files.
    file_names: List of 3D filtered CSV file names.

    Returns:

    object_plane_results: list, object plane points for each trial. Each list element is an array of shape (num_frames, num_keypoints, num_coordinates(x,y,z)).
    hand_plane_results: list, hand plane points for each trial. Each list element is an array of shape (num_frames, num_keypoints, num_coordinates(x,y,z)) .
    '''
    start_frame = round(time_window[0] * frame_rate)
    end_frame = round(time_window[1] * frame_rate)

    # Initialize lists to store results
    object_plane_results = []
    hand_plane_results = []

    for file in file_names:
        # Load 3D filtered data
        f3d_df = pd.read_csv(os.path.join(main_dir, file))
        # Get object plane points
        object_plane_points = f3d_df[[f"{kp}_x" for kp in object_points] +
                                    [f"{kp}_y" for kp in object_points] + 
                                    [f"{kp}_z" for kp in object_points]].values.reshape(-1, 3, 3)[start_frame:end_frame]
        object_plane_points = object_plane_points.transpose(0, 2, 1) # Transpose to shape (num_frames, num_keypoints, num_coordinates(x,y,z))
        object_plane_results.append(object_plane_points)
        # Get palm and wrist points
        hand_plane_points = f3d_df[[f"{hp}_x" for hp in hand_points] + 
                                    [f"{hp}_y" for hp in hand_points] + 
                                    [f"{hp}_z" for hp in hand_points]].values.reshape(-1, 3, 3)[start_frame:end_frame]
        hand_plane_points = hand_plane_points.transpose(0, 2, 1)
        hand_plane_results.append(hand_plane_points)

    return object_plane_results, hand_plane_results

# Get normals and centroids
def compute_plane_normals_and_centroids_per_frame(plane_points):
    '''
    Arguments:
    plane_points: Array of shape (num_frames, num_keypoints, num_coordinates(x,y,z)).
    Returns:
    normals: Array of shape (num_frames, 3), normal vectors for each frame.
    centroids: Array of shape (num_frames, 3), centroids for each frame
    '''
    normals = []
    centroids = []
    for i in range(plane_points.shape[0]): # Iterate through frames
        vec1 = plane_points[i, 1, :] - plane_points[i, 0, :]
        vec2 = plane_points[i, 2, :] - plane_points[i, 0, :]
        normal = np.cross(vec1, vec2)
        normal /= np.linalg.norm(normal)
        normals.append(normal)
        centroid = np.mean(plane_points[i, :, :], axis=0)
        centroids.append(centroid)

    return np.array(normals), np.array(centroids)

 
def compute_normals_and_centroids_per_trial(plane_results):
    '''
    Arguments:

    plane_results: List of arrays containing plane points for each trial. For each trial, the array shape is (num_frames, num_keypoints, num_coordinates(x,y,z)).
    
    Returns:
    hand_plane_normals: Array of shape (num_trials, 3), averaged normal vectors across frames for each trial.
    hand_plane_centroids: Array of shape (num_trials, 3), averaged centroids across framesfor each trial.
    '''
    # Compute hand normals and centroids
    plane_normals = []
    plane_centroids = []

    for idx, plane_points in enumerate(plane_results):
        # 1. Check if array has 0 elements
        if plane_points.size == 0:
            # print(f"Plane points empty for file index {idx}, skipping.")
            plane_normals.append(np.array([np.nan, np.nan, np.nan])) # Append NaN to keep index alignment
            plane_centroids.append(np.array([np.nan, np.nan, np.nan]))
            continue

        # Compute per-frame normals and centroids
        normals, centroids = compute_plane_normals_and_centroids_per_frame(plane_points)

        # 2. Take the average (ignoring NaNs)
        if normals.size > 0 and not np.all(np.isnan(normals)):
            mean_normal = np.nanmean(normals, axis=0)
        else:
            # Handle the empty case (e.g., assign NaN or a default vector)
            mean_normal = np.full(3, np.nan) 

        if centroids.size > 0 and not np.all(np.isnan(centroids)):
            mean_centroid = np.nanmean(centroids, axis=0)
        else:
            mean_centroid = np.full(3, np.nan)

        # 3. Re-normalize the averaged normal vector
        # The arithmetic mean of unit vectors is not a unit vector.
        norm_magnitude = np.linalg.norm(mean_normal)
        if norm_magnitude > 1e-6: # Avoid division by zero
            mean_normal /= norm_magnitude

        plane_normals.append(mean_normal)
        plane_centroids.append(mean_centroid)

    # Convert lists to arrays for easier analysis later
    plane_normals = np.array(plane_normals)
    plane_centroids = np.array(plane_centroids)

    return plane_normals, plane_centroids

def get_hand_object_angles_and_distances(hand_plane_normals, hand_plane_centroids, object_plane_normals, object_plane_centroids):
    
    # Compute mean object normal and centroid across all trials (for reference)
    object_normal_mean = np.nanmean(object_plane_normals, axis=0)
    object_centroid_mean = np.nanmean(object_plane_centroids, axis=0)

    # Compute angles between hand normals and the object reference normals
    angles_deg_list = []
    # Compute centroid distances between hand centroids and object reference centroids
    centroid_distances = []

    for normal, centroid in zip(hand_plane_normals, hand_plane_centroids):
        # 1. Compute angle between hand normal and object normal
        # Dot product: a . b = |a||b|cos(theta) -> cos(theta) = a . b (since normalized)
        dot_product = np.dot(normal, object_normal_mean)
        
        # Clip to handle floating point errors slightly outside [-1, 1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle in degrees (0 to 180)
        angles_deg = np.degrees(np.arccos(dot_product))

        angles_deg_list.append(angles_deg)

        # 2. Compute centroid distance
        distance = np.linalg.norm(centroid - object_centroid_mean)
        centroid_distances.append(distance)

    return np.array(angles_deg_list), np.array(centroid_distances)


def plot_hand_object_analysis(object_plane_centroids, hand_plane_centroids, 
                              object_plane_normals, hand_plane_normals, 
                              angles_deg, centroid_distances, 
                              session_name="Analysis Results",
                              step=10,  # Plot every 10th trial
                              save_path=None):

    fig = plt.figure(figsize=(12, 10))

    # --- 1. Plot Centroids (3D) ---
    # Downsample for visual clarity (slicing [::step])
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    ax1.scatter(object_plane_centroids[::step, 0], object_plane_centroids[::step, 1], object_plane_centroids[::step, 2], 
               color='k', label='Object', s=20, alpha=0.6)
    ax1.scatter(hand_plane_centroids[::step, 0], hand_plane_centroids[::step, 1], hand_plane_centroids[::step, 2], 
               color='b', label='Hand', s=20, alpha=0.6)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Hand and Object Centroids\n(Subsampled 1:{step})')
    ax1.legend()

    # --- 2. Plot Normals (3D Quiver) ---
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    
    # Slice the data first
    obj_norm_sub = object_plane_normals[::step]
    hand_norm_sub = hand_plane_normals[::step]
    
    # Create origin arrays matching the sliced length
    zeros_obj = np.zeros(len(obj_norm_sub))
    zeros_hand = np.zeros(len(hand_norm_sub))

    # Quiver with downsampled data
    ax2.quiver(zeros_obj, zeros_obj, zeros_obj, 
               obj_norm_sub[:, 0], obj_norm_sub[:, 1], obj_norm_sub[:, 2], 
               color='k', length=0.5, normalize=True, label='Object Normals', alpha=0.5)

    ax2.quiver(zeros_hand, zeros_hand, zeros_hand, 
               hand_norm_sub[:, 0], hand_norm_sub[:, 1], hand_norm_sub[:, 2], 
               color='b', length=0.5, normalize=True, label='Hand Normals', alpha=0.5)

    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Normal Vectors (Subsampled 1:{step})')

    # --- 3. Histograms (Use ALL data, no downsampling) ---
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(centroid_distances, bins=20, color='salmon', edgecolor='black')
    ax3.set_title('Centroid Distances (Hand vs Object)')
    ax3.set_xlabel('Distance (units)')
    ax3.set_ylabel('Frequency')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(angles_deg, bins=20, color='skyblue', edgecolor='black')
    ax4.set_xlim(0, 180)
    ax4.set_title('Angles between Normals')
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('Frequency')

    plt.suptitle(session_name, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    plt.show()
# -----------------------------------------------


# ------------ Main execution  --------------
session_name = "2025-12-18"
ap_dir = os.path.join(analysis_dir, session_name, "anipose")
filtered3d_dir = os.path.join(ap_dir, "pose_3d_filter")
f3d_files = sorted(os.listdir(filtered3d_dir))

# Get object and hand plane results from 3D filtered data
object_plane_results, hand_plane_results = get_object_hand_results(filtered3d_dir, f3d_files)
# plane_points shape: (num_trials, num_frames, num_keypoints, num_coordinates(x,y,z))

# Compute normals and centroids of object planes
object_plane_normals, object_plane_centroids = compute_normals_and_centroids_per_trial(object_plane_results)

# Compute normals and centroids of hand planes
hand_plane_normals, hand_plane_centroids = compute_normals_and_centroids_per_trial(hand_plane_results)

# Compute angles and distances
angles_deg, centroid_distances = get_hand_object_angles_and_distances(hand_plane_normals, hand_plane_centroids, object_plane_normals, object_plane_centroids)

# Plot results
save_path = os.path.join(analysis_dir, session_name, "figures", "hand_object_analysis.png")

plot_hand_object_analysis(
    object_plane_centroids, hand_plane_centroids, 
    object_plane_normals, hand_plane_normals, 
    angles_deg, centroid_distances,
    step =10, # Plot every 20th trial
    session_name=session_name, 
    save_path=save_path)

