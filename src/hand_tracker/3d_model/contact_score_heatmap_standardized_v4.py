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

def get_smooth_bone_weights(mesh, canonical_kps, bone_list, sigma=8.0):
    """
    Assigns each vertex a weight for EVERY bone based on distance.
    sigma: controls the smoothness (higher = more blending).
    """
    num_vertices = len(mesh.vertices)
    num_bones = len(bone_list)
    weights = np.zeros((num_vertices, num_bones))
    
    kps_arr = {k: np.array(v) for k, v in canonical_kps.items()}

    for b_idx, (prox, dist, b_type) in enumerate(bone_list):
        p_p, p_d = kps_arr[prox], kps_arr[dist]
        bone_vec = p_d - p_p
        mag_sq = np.dot(bone_vec, bone_vec)
        
        # Calculate distance of every vertex to this bone segment
        v_pts = mesh.vertices - p_p
        t = np.clip(np.dot(v_pts, bone_vec) / mag_sq, 0, 1)
        projection = p_p + np.outer(t, bone_vec)
        dists = np.linalg.norm(mesh.vertices - projection, axis=1)
        
        # Gaussian weighting: influence decreases with distance
        weights[:, b_idx] = np.exp(-(dists**2) / (2 * sigma**2))

    # Normalize weights so they sum to 1.0 for each vertex
    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]
    
    return weights

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

def pose_mesh_lbs(mesh, weights, bone_list, canonical_kps, trial_kps, palm_loop):
    """
    Deforms the hand mesh using Linear Blend Skinning.
    mesh: The canonical STL (trimesh object)
    weights: (N_vertices, N_bones) soft weight matrix
    """
    num_vertices = len(mesh.vertices)
    num_bones = len(bone_list)
    
    # 1. Prepare Keypoints as Numpy Arrays
    c_kps = {k: np.array(v) for k, v in canonical_kps.items()}
    t_kps = {k: np.array(v) for k, v in trial_kps.items()}
    
    # 2. Pre-calculate Palm and Trial Normals for axial alignment
    def get_normal(kpts):
        v1 = kpts["Middle_MCP"] - ((kpts["Wrist_R"] + kpts["Wrist_U"]) / 2.0)
        v2 = kpts["Index_MCP"] - kpts["Small_MCP"]
        n = np.cross(v1, v2)
        return n / np.linalg.norm(n)

    can_normal = get_normal(c_kps)
    trial_normal = get_normal(t_kps)

    # 3. Calculate 4x4 Transformation Matrices for every bone
    bone_transforms = []
    
    for prox, dist, b_type in bone_list:
        if b_type == "palm":
            # Use Procrustes for the palm base
            src_palm = np.array([c_kps[k] for k in palm_loop])
            tgt_palm = np.array([t_kps[k] for k in palm_loop])
            R_palm, t_off = get_rigid_transform(src_palm, tgt_palm)
            
            # Convert to 4x4 matrix
            M = np.eye(4)
            M[:3, :3] = R_palm
            M[:3, 3] = t_off
            bone_transforms.append(M)
            
        else:
            # Use Look-At Basis Matrix for fingers
            cp, cd = c_kps[prox], c_kps[dist]
            tp, td = t_kps[prox], t_kps[dist]
            
            # Source Frame
            v_can = (cd - cp) / np.linalg.norm(cd - cp)
            x_can = np.cross(v_can, can_normal); x_can /= np.linalg.norm(x_can)
            y_can = np.cross(x_can, v_can)
            M_src = np.eye(4)
            M_src[:3, :3] = np.column_stack([x_can, y_can, v_can])
            M_src[:3, 3] = cp

            # Target Frame
            v_tri = (td - tp) / np.linalg.norm(td - tp)
            x_tri = np.cross(v_tri, trial_normal); x_tri /= np.linalg.norm(x_tri)
            y_tri = np.cross(x_tri, v_tri)
            M_tgt = np.eye(4)
            M_tgt[:3, :3] = np.column_stack([x_tri, y_tri, v_tri])
            M_tgt[:3, 3] = tp

            # The relative transform that moves a vertex from canonical bone space to trial bone space
            # Formula: Target * Inverse(Source)
            bone_transforms.append(M_tgt @ np.linalg.inv(M_src))

    # 4. Perform Linear Blend Skinning
    # Final_Pos = Sum(Weight_i * (M_i * Initial_Pos))
    
    # Convert vertices to homogeneous coordinates (N, 4)
    homog_vertices = np.hstack([mesh.vertices, np.ones((num_vertices, 1))])
    new_vertices_homog = np.zeros((num_vertices, 4))

    for b_idx in range(num_bones):
        # Apply this bone's transform to ALL vertices
        transformed_pts = (bone_transforms[b_idx] @ homog_vertices.T).T
        
        # Add to the running sum, weighted by the specific influence of this bone on each vertex
        w = weights[:, b_idx][:, np.newaxis]
        new_vertices_homog += w * transformed_pts

    # 5. Return as a new trimesh
    return trimesh.Trimesh(vertices=new_vertices_homog[:, :3], faces=mesh.faces)


# --- 4. MAIN ANALYSIS PIPELINE ---

# A. Pre-calculate mapping (Identity)
vertex_map, bones = get_bone_mapping(canonical_hand_mesh, canonical_hand_kpts, finger_chains, palm_loop)
weights = get_smooth_bone_weights(canonical_hand_mesh, canonical_hand_kpts, bones, sigma=4.0)

# B. Deform Hand Mesh
posed_hand_mesh = pose_mesh_lbs(canonical_hand_mesh, weights, bones, canonical_hand_kpts, trial_kpts, palm_loop)

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

    # Static View for Canonical Hand (Ventral View)
    ax2.view_init(elev=10, azim=-85, roll=-180)

    # Set limits to fit the canonical hand
    ax2.set_xlim(-50, 51); ax2.set_ylim(0, 60); ax2.set_zlim(-90, 30)
    ax2.set_axis_off()
    ax2.set_title("Standardized Canonical Heatmap")

    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
    cb_ax = fig.add_axes([0.48, 0.2, 0.02, 0.6])
    fig.colorbar(sm, cax=cb_ax, label='Cumulative Contact Score')

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