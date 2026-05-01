from pathlib import Path
import json
import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- CONFIGURATION ---
# Canonical data paths
CANONICAL_HAND_STL_PATH = "/media/yiting/NewVolume/Data/Hand_anatomy/MRI/Neo Hand Segmentation v1 decimated 10.stl"
CANONICAL_HAND_KPTS_PATH = "/home/yiting/Documents/GitHub/hand_tracking/configs/hand_keypoint_map.json"
# Trial data paths
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
STL_ROOT = Path("/media/yiting/NewVolume/Data/Shapes/shapes_stl")
session_name = "2025-12-09"
trial_name = "2025-12-09_09-01-20"
FRAME_NUMBER = 300

DISTANCE_THRESHOLDS_MM = np.arange(-2, 4, 0.025)

finger_chains = {
    "Small": ["Small_Tip", "Small_DIP", "Small_PIP", "Small_MCP"],
    "Ring": ["Ring_Tip", "Ring_DIP", "Ring_PIP", "Ring_MCP"],
    "Middle": ["Middle_Tip", "Middle_DIP", "Middle_PIP", "Middle_MCP"],
    "Index": ["Index_Tip", "Index_DIP", "Index_PIP", "Index_MCP"],
    "Thumb": ["Thumb_Tip", "Thumb_IP", "Thumb_MCP"]
}
palm_loop = ["Small_MCP", "Ring_MCP", "Middle_MCP", "Index_MCP", "Thumb_MCP", "Thumb_CMC", "Wrist_R", "Wrist_U"]


# --- 1. LOAD CANONICAL DATA ---
canonical_hand_mesh = trimesh.load(CANONICAL_HAND_STL_PATH)
with open(CANONICAL_HAND_KPTS_PATH, "r") as f:
    canonical_hand_kpts = json.load(f)

def get_bone_mapping(mesh, canonical_kps, finger_chains, palm_loop):
    vertex_bone_assignments = []
    bone_list = [] # List of (proximal, distal, type)
    
    # 1. Add Finger Bones
    for name, chain in finger_chains.items():
        for i in range(len(chain)-1):
            bone_list.append((chain[i+1], chain[i], "finger")) 
            
    # 2. Add Palm segments to capture palm vertices
    # We connect the wrist markers to the outer MCPs to form a "frame"
    palm_connections = [("Wrist_R", "Index_MCP"), ("Wrist_U", "Small_MCP"), ("Wrist_R", "Thumb_CMC")]
    for prox, dist in palm_connections:
        bone_list.append((prox, dist, "palm"))

    # Convert kps to arrays once for speed
    kps_arr = {k: np.array(v) for k, v in canonical_kps.items()}
            
    for v in mesh.vertices:
        min_dist = float('inf')
        best_bone_idx = 0
        for idx, (prox, dist, b_type) in enumerate(bone_list):
            p_p, p_d = kps_arr[prox], kps_arr[dist]
            bone_vec = p_d - p_p
            mag_sq = np.dot(bone_vec, bone_vec)
            if mag_sq < 1e-6: continue 
            
            t = np.clip(np.dot(v - p_p, bone_vec) / mag_sq, 0, 1)
            projection = p_p + t * bone_vec
            d = np.linalg.norm(v - projection)
            
            if d < min_dist:
                min_dist = d
                best_bone_idx = idx
        vertex_bone_assignments.append(best_bone_idx)
        
    return np.array(vertex_bone_assignments), bone_list

# --- 2. LOAD TRIAL DATA ---
def get_xyz(name, frame_data):
    return np.array([frame_data[f"{name}_x"], frame_data[f"{name}_y"], frame_data[f"{name}_z"]])

# Load 3D pose data
pose_3d_dir = ANALYSIS_ROOT / session_name / 'anipose' / 'pose_3d_filter'
df = pd.read_csv(pose_3d_dir / f'{trial_name}_f3d.csv')
f = df.iloc[FRAME_NUMBER] # The 'f' variable
trial_kpts = {name: get_xyz(name, f) for name in canonical_hand_kpts.keys()}

# Load object and orientation info
log_path = RAW_DATA_ROOT / session_name / 'trial_logs' / f'{trial_name}_log.json'
with open(log_path, 'r') as file:
    log_data = json.load(file)
shape_id = log_data.get("shape_id", "unknown_0")
obj_id, orientation = shape_id.split("_")[0], shape_id.split("_")[-1]

# Load Object STL and Configs
obj_mesh = trimesh.load(STL_ROOT / f'{obj_id}.stl')
with open(Path("/home/yiting/Documents/GitHub/hand_tracking/configs/obj_coordinates.json"), 'r') as file:
    obj_configs = json.load(file)


# --- 3. DEFORM CANONICAL TO NEW POSE ---

def get_rigid_transform(src, tgt):
    mask = ~np.isnan(src).any(axis=1) & ~np.isnan(tgt).any(axis=1)
    A, B = src[mask], tgt[mask]
    centroid_A, centroid_B = np.mean(A, axis=0), np.mean(B, axis=0)
    H = (A - centroid_A).T @ (B - centroid_B)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    return R, centroid_B - R @ centroid_A

# For each bone, calculate the rigid transform from canonical to trial, then apply to assigned vertices.
def pose_mesh(mesh, vertex_map, bone_list, canonical_kps, trial_kps, palm_loop):
    new_vertices = np.copy(mesh.vertices)
    
    # FIX 1: Convert dictionary values from lists to numpy arrays immediately
    c_kps = {k: np.array(v) for k, v in canonical_kps.items()}
    t_kps = {k: np.array(v) for k, v in trial_kps.items()}
    
    # 1. Calculate the Trial Palm Normal (The "Target Ventral" direction)
    # Using the new 't_kps' dictionary
    v1 = t_kps["Middle_MCP"] - ((t_kps["Wrist_R"] + t_kps["Wrist_U"]) / 2.0)
    v2 = t_kps["Index_MCP"] - t_kps["Small_MCP"]
    trial_normal = np.cross(v1, v2)
    trial_normal /= np.linalg.norm(trial_normal)

    # 2. Calculate the Canonical Palm Normal (The "Source Ventral" direction)
    # Using the new 'c_kps' dictionary
    cv1 = c_kps["Middle_MCP"] - ((c_kps["Wrist_R"] + c_kps["Wrist_U"]) / 2.0)
    cv2 = c_kps["Index_MCP"] - c_kps["Small_MCP"]
    can_normal = -np.cross(cv1, cv2)
    can_normal /= np.linalg.norm(can_normal)

    # --- Handle Palm (Procrustes) ---
    palm_indices = [i for i, b in enumerate(bone_list) if b[2] == "palm"]
    src_palm = np.array([c_kps[k] for k in palm_loop])
    tgt_palm = np.array([t_kps[k] for k in palm_loop])
    R_palm, t_palm = get_rigid_transform(src_palm, tgt_palm)
    
    palm_mask = np.isin(vertex_map, palm_indices)
    c_palm_centroid = np.mean(src_palm, axis=0)
    t_palm_centroid = np.mean(tgt_palm, axis=0)
    
    new_vertices[palm_mask] = (new_vertices[palm_mask] - c_palm_centroid) @ R_palm.T + t_palm_centroid

    # --- Handle Fingers with Axial Constraint ---
    for idx, (prox, dist, b_type) in enumerate(bone_list):
        if b_type != "finger": continue
        
        c_p, c_d = np.array(canonical_kps[prox]), np.array(canonical_kps[dist])
        t_p, t_d = np.array(trial_kps[prox]), np.array(trial_kps[dist])
        
        # SOURCE FRAME
        v_can = (c_d - c_p) / np.linalg.norm(c_d - c_p)
        # Use canonical normal as the "up" reference
        # Orthogonalize to get a clean basis
        x_can = np.cross(v_can, can_normal)
        y_can = np.cross(x_can, v_can)
        M_src = np.column_stack([x_can, y_can, v_can])

        # TARGET FRAME
        v_tri = (t_d - t_p) / np.linalg.norm(t_d - t_p)
        # Use trial normal as the "up" reference
        x_tri = np.cross(v_tri, trial_normal)
        y_tri = np.cross(x_tri, v_tri)
        M_tgt = np.column_stack([x_tri, y_tri, v_tri])

        # ROTATION MATRIX: Moves M_src to M_tgt
        R_bone = M_tgt @ M_src.T
        
        mask = (vertex_map == idx)
        if not np.any(mask): continue
        
        # Apply transform: Translate to origin, Rotate, Translate to target joint
        pts = new_vertices[mask] - c_p
        new_vertices[mask] = (pts @ R_bone.T) + t_p
        
    return trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)


# --- 4. MAIN ANALYSIS PIPELINE ---

# A. Pre-calculate mapping (Identity)
vertex_map, bones = get_bone_mapping(canonical_hand_mesh, canonical_hand_kpts, finger_chains, palm_loop)

# B. Deform Hand Mesh
posed_hand_mesh = pose_mesh(canonical_hand_mesh, vertex_map, bones, canonical_hand_kpts, trial_kpts, palm_loop)

# C. Load Object and Calculate Scores
# Align STL to tracked markers
obj_map = obj_configs["orientations"][orientation]
src_obj = np.array(list(obj_map.values()))
tgt_obj = np.array([get_xyz(name, f) for name in obj_map.keys()])
R, t = get_rigid_transform(src_obj, tgt_obj)

matrix = np.eye(4)
matrix[:3, :3] = R
matrix[:3, 3] = t
obj_mesh.apply_transform(matrix)

# Initialize 'obj_tree' for distance queries
obj_tree = KDTree(obj_mesh.vertices)

# Query distances from posed hand vertices to object surface
dists, _ = obj_tree.query(posed_hand_mesh.vertices)
vertex_scores = np.sum([dists <= t for t in DISTANCE_THRESHOLDS_MM], axis=0)

# # --- 5. VISUALIZATION  ---
def visualize_dual_view(posed_mesh, canonical_mesh, obj_mesh, vertex_scores, thresholds, trial_name):
    """
    Generates a side-by-side comparison:
    Left: Posed hand with object interaction.
    Right: Standardized heatmap on the flat canonical mesh.
    """
    fig = plt.figure(figsize=(20, 10))
    norm = plt.Normalize(vmin=0, vmax=len(thresholds))

    # --- LEFT SUBPLOT: POSED INTERACTION ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Calculate face colors for posed mesh
    face_colors_posed = cm.YlOrRd(norm(vertex_scores[posed_mesh.faces].mean(axis=1)))
    
    # Render Posed Hand
    poly_posed = Poly3DCollection(posed_mesh.vertices[posed_mesh.faces], 
                                  facecolors=face_colors_posed, edgecolor='none', alpha=0.9, shade=True)
    ax1.add_collection3d(poly_posed)
    
    # Render Object (X-Ray effect)
    ax1.plot_trisurf(obj_mesh.vertices[:, 0], obj_mesh.vertices[:, 1], obj_mesh.vertices[:, 2],
                     triangles=obj_mesh.faces, color='gray', alpha=0.2, edgecolor='none', shade=True)

    # Dynamic Scaling for Left View
    all_v = np.vstack([posed_mesh.vertices, obj_mesh.vertices])
    mid = all_v.mean(axis=0); max_r = (all_v.max(axis=0) - all_v.min(axis=0)).max() / 2.0
    ax1.set_xlim(mid[0]-max_r, mid[0]+max_r); ax1.set_ylim(mid[1]-max_r, mid[1]+max_r); ax1.set_zlim(mid[2]-max_r, mid[2]+max_r)
    
    ax1.set_axis_off()
    ax1.view_init(elev=20, azim=45)
    ax1.set_title(f"Posed Interaction: {trial_name}", fontsize=15)

    # --- RIGHT SUBPLOT: CANONICAL HEATMAP ---
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Transfer same vertex scores to the flat canonical mesh faces
    face_colors_canonical = cm.YlOrRd(norm(vertex_scores[canonical_mesh.faces].mean(axis=1)))
    
    # Render Canonical Flat Hand
    poly_can = Poly3DCollection(canonical_mesh.vertices[canonical_mesh.faces], 
                                facecolors=face_colors_canonical, edgecolor='none', alpha=1.0, shade=True)
    ax2.add_collection3d(poly_can)

    # Static View for Canonical Hand (Top-Down)
    ax2.set_xlim(-50, 50); ax2.set_ylim(-40, 60); ax2.set_zlim(-10, 10)
    ax2.view_init(elev=90, azim=-90)
    ax2.set_axis_off()
    ax2.set_title("Standardized Canonical Heatmap", fontsize=15)

    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
    cb_ax = fig.add_axes([0.48, 0.2, 0.02, 0.6]) # Vertical colorbar in middle
    fig.colorbar(sm, cax=cb_ax, label='Cumulative Contact Score (Depth)')

    plt.tight_layout()
    plt.show()

visualize_dual_view(posed_hand_mesh, canonical_hand_mesh, obj_mesh, vertex_scores, DISTANCE_THRESHOLDS_MM, trial_name)

# # --- 5. VISUALIZATION ON CANONICAL (FLAT) MESH ---
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Map scores to the ORIGINAL FLAT STL faces
# norm = plt.Normalize(vmin=0, vmax=len(DISTANCE_THRESHOLDS_MM))
# face_colors = cm.YlOrRd(norm(vertex_scores[canonical_hand_mesh.faces].mean(axis=1)))

# poly = Poly3DCollection(canonical_hand_mesh.vertices[canonical_hand_mesh.faces], 
#                         facecolors=face_colors, edgecolor='none', shade=True)
# ax.add_collection3d(poly)

# ax.view_init(elev=90, azim=-90)
# ax.set_axis_off()
# plt.title(f"Contact Heatmap on Canonical STL: {trial_name}")
# plt.show()