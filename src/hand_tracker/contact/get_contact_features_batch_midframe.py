import os
from pathlib import Path
from glob import glob
import json
import pandas as pd
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, KDTree
from hand_tracker.utils.file_io import get_trialname, find_log_or_robot
from hand_tracker.utils.analysis_window import load_window_lookup

'''
This module implements the batch processing pipeline to extract standardized hand contact features 
across multiple trials within a session. It integrates the geometric reconstruction and contact scoring logic 
from the single-trial pipeline. 

Unlike get_contact_features_batch.py (which averages the contact vector across every frame in each
trial's hold window), this variant runs the reconstruction only once, at the middle frame of the
hold window, for much faster processing at the cost of losing the within-window averaging.

The output is a consolidated DataFrame containing trial metadata and the corresponding contact feature vectors 
(n = resolution x resolution). It also saves the generated 3D heatmaps and standardized 2D contact maps for 
each trial in the respective session's reconstruction directory.
'''
# ==========================================
# 0. GLOBAL PATHS & CONFIGURATIONS
# ==========================================
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
STL_ROOT = Path("/media/yiting/NewVolume/Data/Shapes/shapes_stl")
CONFIG_JSON_PATH = Path("/home/yiting/Documents/GitHub/hand_tracking/configs/obj_coordinates.json")

# Morphological Hyperparameters
FINGER_DIAMETER_MM = 8.0
PALM_THICKNESS = 9.0
LW = (FINGER_DIAMETER_MM / 25.4) * 72 

HAND_COLOR = "#FFCC99"
HAND_OPACITY = 0.1 
OBJECT_OPACITY = 0.5 
DISTANCE_THRESHOLDS_MM = np.arange(-3, 6, 0.01)

# Hand Kinematic Connectivity Maps
FINGER_CHAINS = {
    "Small": ["Small_Tip", "Small_DIP", "Small_PIP", "Small_MCP"],
    "Ring": ["Ring_Tip", "Ring_DIP", "Ring_PIP", "Ring_MCP"],
    "Middle": ["Middle_Tip", "Middle_DIP", "Middle_PIP", "Middle_MCP"],
    "Index": ["Index_Tip", "Index_DIP", "Index_PIP", "Index_MCP"],
    "Thumb": ["Thumb_Tip", "Thumb_IP", "Thumb_MCP"]
}
PALM_LOOP = ["Small_MCP", "Ring_MCP", "Middle_MCP", "Index_MCP", "Thumb_MCP", "Thumb_CMC", "Wrist_R", "Wrist_U"]

# Standardized Heatmap Parameters
HEATMAP_RESOLUTION = 64 # Number of pixels along each dimension of the standardized contact map and the resulting contact vector length will be resolution^2 
FINGER_PALM_SPLIT_RATIO = 0.4 # Proportion of the vertical axis allocated to the palm region in the standardized heatmap

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

def load_tracking_df(session_name, trial_name):
    """Loads the full 3D filter coordinates file for a trial (all frames)."""
    pose_3d_dir = ANALYSIS_ROOT / session_name / 'anipose' / 'pose_3d_filter'
    csv_path = pose_3d_dir / f'{trial_name}_f3d.csv'
    return pd.read_csv(csv_path)

def get_marker_configs():
    """Load object marker positions in a stimulus coordinate system from the JSON config file."""
    with open(CONFIG_JSON_PATH, 'r') as file:
        return json.load(file)

def get_trial_entry(trial_name, log_fname):
    with open(log_fname, 'r') as file:
        log_data = json.load(file)
                
    trial_entry = {
        "trial_name": trial_name,
        "shape_id": log_data.get("shape_id", "unknown_0"),
        "correct": log_data.get("has_played_success_tone", False),
        "is_holdshort": log_data.get("object_released", False),
        "is_holdlong": log_data.get("object_held", False)
    }
    return trial_entry
    

# ==========================================
# 2. RIGID TRANSFORMS & GEOMETRY MODULE
# ==========================================
def get_rigid_transform(src, tgt):
    """Calculates R and t such that B = R*A + t (SVD Procrustes Analysis)."""
    mask = ~np.isnan(src).any(axis=1) & ~np.isnan(tgt).any(axis=1)
    A = src[mask]
    B = tgt[mask]
    
    if len(A) < 3:
        raise ValueError("Not enough valid non-NaN target tracking markers to compute alignment.")
    
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t

def extract_keypoint_xyz(frame_row, kp_name):
    """Helper to cleanly parse dynamic scalar keys into 3D vectors."""
    return np.array([frame_row[f"{kp_name}_x"], frame_row[f"{kp_name}_y"], frame_row[f"{kp_name}_z"]])

def compute_palm_normal(frame_row):
    """Calculates standard directional surface normal pointing out of palm."""
    wrist_avg = (extract_keypoint_xyz(frame_row, "Wrist_R") + extract_keypoint_xyz(frame_row, "Wrist_U")) / 2.0
    v1 = extract_keypoint_xyz(frame_row, "Middle_MCP") - wrist_avg
    v2 = extract_keypoint_xyz(frame_row, "Index_MCP") - extract_keypoint_xyz(frame_row, "Small_MCP")
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)


# ==========================================
# 3. DENSE SURFACE SAMPLING MODULE
# ==========================================
def sample_finger_surface(frame_row, normal, diameter=FINGER_DIAMETER_MM):
    """Generates continuous geometric point cloud models along finger segments."""
    surface_points = []
    radius = diameter / 2.0
    
    for name, chain in FINGER_CHAINS.items():
        # 1. Joint Spheres
        for joint_name in chain:
            joint_center = extract_keypoint_xyz(frame_row, joint_name)
            res = 25 if "Tip" in joint_name else 20
            for phi in np.linspace(0, np.pi, res):
                for theta in np.linspace(0, 2*np.pi, res):
                    dx = radius * np.sin(phi) * np.cos(theta)
                    dy = radius * np.sin(phi) * np.sin(theta)
                    dz = radius * np.cos(phi)
                    test_pt = joint_center + np.array([dx, dy, dz])
                    if np.dot(test_pt - joint_center, -normal) > 0:
                        surface_points.append(test_pt)

        # 2. Bone Cylinders
        for i in range(len(chain) - 1):
            p1 = extract_keypoint_xyz(frame_row, chain[i])     
            p2 = extract_keypoint_xyz(frame_row, chain[i+1])   
            v_seg = p2 - p1
            dist_seg = np.linalg.norm(v_seg)
            z_axis = v_seg / dist_seg
            
            ref_vec = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            x_axis = np.cross(ref_vec, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            num_steps = max(int(dist_seg / 0.5), 2)
            num_angles = 24
            for s in np.linspace(0, 1, num_steps):
                center = p1 + s * v_seg
                for angle in np.linspace(0, 2 * np.pi, num_angles):
                    test_pt = center + radius * (np.cos(angle) * x_axis + np.sin(angle) * y_axis)
                    if np.dot(test_pt - center, -normal) > 0:
                        surface_points.append(test_pt)
                        
    return np.array(surface_points)

def sample_palm_surface(all_palm_cloud, hull, normal, num_samples=10000):
    """Uniform area-weighted distribution selection across active palm envelope."""
    simplices = hull.simplices
    def tri_area(p1, p2, p3):
        return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

    areas = np.array([tri_area(*all_palm_cloud[s]) for s in simplices])
    probs = areas / np.sum(areas)
    chosen_indices = np.random.choice(len(simplices), size=num_samples, p=probs)
    
    sampled_points = []
    palm_center = all_palm_cloud.mean(axis=0)

    for idx in chosen_indices:
        tri_pts = all_palm_cloud[simplices[idx]]
        r1, r2 = np.sqrt(np.random.random()), np.random.random()
        pt = (1 - r1) * tri_pts[0] + r1 * (1 - r2) * tri_pts[1] + r1 * r2 * tri_pts[2]
        if np.dot(pt - palm_center, -normal) > 0:
            sampled_points.append(pt)
            
    return np.array(sampled_points)


# ==========================================
# 4. QUANTIFICATION AND FLATTENING MODULE
# ==========================================
def compute_contact_scores(surface_points, obj_tree, thresholds=DISTANCE_THRESHOLDS_MM):
    """Iterates spatial skin locations against target objects over dynamic offsets."""
    distances, _ = obj_tree.query(surface_points)
    scores = np.zeros(len(surface_points))
    for t in thresholds:
        scores += (distances <= t).astype(int)
    return scores, distances

def generate_standardized_hand_map(frame_row, hand_surface, scores, resolution=HEATMAP_RESOLUTION):
    """Unrolls volumetric interaction signatures into canvas maps for comparisons."""
    flat_map = np.zeros((resolution, resolution))
    v_split = int(resolution * FINGER_PALM_SPLIT_RATIO)
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Small"]
    col_width = resolution // len(finger_names)

    # Fingers
    for f_idx, name in enumerate(finger_names):
        chain = FINGER_CHAINS[name]
        joints = [extract_keypoint_xyz(frame_row, j) for j in chain]
        bone_lengths = [np.linalg.norm(joints[i] - joints[i+1]) for i in range(len(joints)-1)]
        total_skeletal_len = sum(bone_lengths)
        u_start, u_end = f_idx * col_width, (f_idx + 1) * col_width

        for pt, score in zip(hand_surface, scores):
            for i in range(len(joints)-1):
                p_distal, p_prox = joints[i], joints[i+1]
                bone_vec = p_distal - p_prox
                bone_unit = bone_vec / np.linalg.norm(bone_vec)
                proj = np.dot(pt - p_prox, bone_unit)
                dist_to_axis = np.linalg.norm((pt - p_prox) - proj * bone_unit)
                
                if 0 <= proj <= np.linalg.norm(bone_vec) and dist_to_axis < 6.0:
                    len_from_mcp = sum(bone_lengths[k] for k in range(i+1, len(joints)-1)) + proj
                    norm_v = len_from_mcp / total_skeletal_len
                    v_idx = v_split + int(norm_v * (resolution - v_split - 1))
                    flat_map[v_idx, u_start:u_end] = np.maximum(flat_map[v_idx, u_start:u_end], score)

    # Palm
    p_idx_mcp = extract_keypoint_xyz(frame_row, "Index_MCP")
    p_sml_mcp = extract_keypoint_xyz(frame_row, "Small_MCP")
    p_wri_r   = extract_keypoint_xyz(frame_row, "Wrist_R")
    p_wri_u   = extract_keypoint_xyz(frame_row, "Wrist_U")
    v_axis = ((p_idx_mcp + p_sml_mcp)/2) - ((p_wri_r + p_wri_u)/2)
    u_axis = p_sml_mcp - p_idx_mcp

    for pt, score in zip(hand_surface, scores):
        palm_center = (p_idx_mcp + p_sml_mcp + p_wri_r + p_wri_u) / 4.0
        if np.linalg.norm(pt - palm_center) < 25.0:
            u_palm = np.dot(pt - p_idx_mcp, u_axis) / np.dot(u_axis, u_axis)
            v_palm = np.dot(pt - p_wri_r, v_axis) / np.dot(v_axis, v_axis)
            if 0 <= u_palm <= 1 and 0 <= v_palm <= 1:
                u_idx = int(u_palm * (resolution - 1))
                v_idx = int(v_palm * (v_split - 1))
                flat_map[v_idx, u_idx] = max(flat_map[v_idx, u_idx], score)

    return flat_map


# ==========================================
# 5. VISUALIZATION AND SAVE EXPORTERS
# ==========================================
def save_3d_heatmap(hand_surface_points, scores, v_mesh, faces_mesh, hull_faces, top_pts, bottom_pts, output_path):
    """Generates and saves the 3D projection graphic plot."""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Palm Mesh Layout
    palm_volume = Poly3DCollection(hull_faces, facecolors=HAND_COLOR, edgecolors='k', linewidths=0.2, alpha=HAND_OPACITY, zorder=5)
    ax.add_collection3d(palm_volume)
    
    # Object Mesh Model
    ax.plot_trisurf(v_mesh[:,0], v_mesh[:,1], v_mesh[:,2], triangles=faces_mesh, color='gray', alpha=OBJECT_OPACITY, edgecolor='none', zorder=1)
    
    # Dense Color Heatmap Scatter
    ax.scatter(hand_surface_points[:,0], hand_surface_points[:,1], hand_surface_points[:,2], c=scores, cmap='YlOrRd', s=2, alpha=0.9, edgecolors='none')
    
    ax.set_title("3D Hand Reconstruction: Contact Score Heatmap")
    ax.set_axis_off()
    
    all_pts = np.vstack([v_mesh, top_pts, bottom_pts])
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    
    ax.computed_zorder = False
    ax.view_init(elev=3, azim=82, roll=-10)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_flat_heatmap(standardized_heatmap, output_path):
    """Generates and saves the 2D canonical heatmap plot."""
    fig = plt.figure(figsize=(6, 8))
    plt.imshow(standardized_heatmap, cmap='YlOrRd', origin='lower', aspect='auto')
    plt.axhline(y=int(HEATMAP_RESOLUTION*FINGER_PALM_SPLIT_RATIO), color='black', linestyle='--', label='MCP Line')
    plt.title("Standardized Hand Contact Map")
    plt.xlabel("Fingers (Thumb $\\rightarrow$ Small)")
    plt.ylabel("Proximal $\\rightarrow$ Distal")
    plt.colorbar(label='Contact Score')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ==========================================
# 6. PIPELINE CONTROLLER EXECUTION
# ==========================================
def compute_frame_heatmap(frame_row, dot_configs, orientation, mesh_original):
    """Runs the geometric reconstruction and contact-scoring pipeline for a single
    tracked frame. Returns the standardized 2D contact heatmap plus the intermediate
    pieces needed if this frame is later chosen for visualization."""
    mesh = mesh_original.copy()

    dot_map = dot_configs["orientations"][orientation]
    src_dots = np.array(list(dot_map.values()))
    tgt_dots = np.array([extract_keypoint_xyz(frame_row, name) for name in dot_map.keys()])

    R, t = get_rigid_transform(src_dots, tgt_dots)
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = t
    mesh.apply_transform(matrix)

    obj_tree = KDTree(mesh.vertices)

    # 2. Geometric Core Calculations
    normal = compute_palm_normal(frame_row)
    palm_pts = np.array([extract_keypoint_xyz(frame_row, pt) for pt in PALM_LOOP])

    top_pts = palm_pts + (normal * (PALM_THICKNESS / 2.0))
    bottom_pts = palm_pts - (normal * (PALM_THICKNESS / 2.0))
    all_palm_cloud = np.vstack([top_pts, bottom_pts, palm_pts.mean(axis=0)])
    hull = ConvexHull(all_palm_cloud)
    hull_faces = [all_palm_cloud[s] for s in hull.simplices]

    # 3. Dense Point Sampling & Cumulative Distance Scoring
    finger_surface_points = sample_finger_surface(frame_row, normal)
    palm_surface_points = sample_palm_surface(all_palm_cloud, hull, normal)
    hand_surface_points = np.vstack([finger_surface_points, palm_surface_points])

    scores, dists = compute_contact_scores(hand_surface_points, obj_tree)
    standardized_heatmap = generate_standardized_hand_map(frame_row, hand_surface_points, scores)

    render_extras = {
        "mesh": mesh,
        "hand_surface_points": hand_surface_points,
        "scores": scores,
        "hull_faces": hull_faces,
        "top_pts": top_pts,
        "bottom_pts": bottom_pts,
    }
    return standardized_heatmap, render_extras


def process_single_trial_pipeline(session_name, trial_name, log_fname, start_frame, end_frame):
    """Runs the reconstruction pipeline once, at the middle frame of the hold window,
    instead of averaging across every frame (much faster, less precise)."""
    middle_frame = (start_frame + end_frame) // 2
    print(f"-> Commencing Analysis Pipeline: Trial '{trial_name}' | Hold window [{start_frame}-{end_frame}] | Mid frame [{middle_frame}]")

    # 1. Load Configurations and Tracking Points
    dot_configs = get_marker_configs()
    obj_id, orientation, shape_id = load_trial_metadata(log_fname)

    stl_path = STL_ROOT / f'{obj_id}.stl'
    mesh_original = trimesh.load(stl_path)

    tracking_df = load_tracking_df(session_name, trial_name)
    middle_frame = min(middle_frame, len(tracking_df) - 1)
    frame_row = tracking_df.iloc[middle_frame]

    standardized_heatmap, render_extras = compute_frame_heatmap(frame_row, dot_configs, orientation, mesh_original)

    # 4. Export Maps and Heatmaps
    recon_dir = ANALYSIS_ROOT / session_name / 'reconstructions' / trial_name
    recon_dir.mkdir(parents=True, exist_ok=True)

    img_3d_path = recon_dir / f'contact_scores_{trial_name}_midframe.png'
    save_3d_heatmap(
        render_extras["hand_surface_points"], render_extras["scores"],
        render_extras["mesh"].vertices, render_extras["mesh"].faces,
        render_extras["hull_faces"], render_extras["top_pts"],
        render_extras["bottom_pts"], img_3d_path,
    )

    img_flat_path = recon_dir / f'contact_scores_standardized_heatmap_{trial_name}_midframe_v1.png'
    save_flat_heatmap(standardized_heatmap, img_flat_path)

    # Get trial entry and flat feature vector
    trial_entry = get_trial_entry(trial_name, log_fname)
    trial_entry["contact_vector"] = standardized_heatmap.ravel().tolist()

    print(f"   Successfully generated output assets inside: {recon_dir}\n")
    return trial_entry

def batch_process_session(session_name, trial_names, log_fnames, window_lookup):
    """Processes multiple trial configurations recorded across an identical session folder."""
    print(f"=== Initiating Batch Processing Session: {session_name} ===")
    integrated_rows = []
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

        try:
            trial_entry = process_single_trial_pipeline(session_name, trial_name, log_fname, start_frame, end_frame)
            integrated_rows.append(trial_entry)
        except Exception as e:
            print(f"❌ Error occurred while processing Trial: '{trial_name}'. Details: {e}\n")
            continue
            
    if skipped_no_window:
        print(f"Skipped {skipped_no_window} trials with no hold window on record.")
    print("=== Batch Processing Sequence Completed ===")
    return integrated_rows


# ==========================================
# 7. MAIN ROUTINE ENTRY POINT
# ==========================================
if __name__ == "__main__":

    session_names = ["2025-11-19", "2025-12-04", "2025-12-16", "2025-12-17"]
    
    # session_names = ["2025-08-19", "2025-08-22", "2025-11-19", "2025-11-20", "2025-12-04",
    #                     "2025-12-08", "2025-12-09", "2025-12-16", "2025-12-17", "2025-12-18"]
    
    for session_name in session_names:
        feature_dir = os.path.join(ANALYSIS_ROOT, session_name, "features")
        log_dir = os.path.join(RAW_DATA_ROOT, session_name, "trial_logs")

        window_lookup = load_window_lookup(session_name)
        if window_lookup is None:
            print(f"Warning: no min_holding_window.csv found for {session_name}, skipping session.")
            continue

        feature_fnames = sorted(glob(os.path.join(feature_dir, "*.csv")))
        log_fnames = find_log_or_robot(feature_fnames, log_dir)
        trial_names = [get_trialname(f) for f in feature_fnames]

        # Execute batch pipeline
        integrated_rows = batch_process_session(session_name, trial_names, log_fnames, window_lookup)

        if integrated_rows:
            master_df = pd.DataFrame(integrated_rows, columns=["trial_name", "shape_id", "correct", "is_holdshort", "is_holdlong", "contact_vector"])
            
            master_df['contact_vector'] = master_df['contact_vector'].apply(lambda x: str(x))
            
            contact_output_dir = ANALYSIS_ROOT / session_name / "contact"
            contact_output_dir.mkdir(parents=True, exist_ok=True)
            output_csv_path = contact_output_dir / f"contact_features_{session_name}_midframe.csv"
            
            # Save to CSV
            master_df.to_csv(output_csv_path, index=False)
            print(f"✅ Successfully saved integrated contact features CSV: {output_csv_path}")