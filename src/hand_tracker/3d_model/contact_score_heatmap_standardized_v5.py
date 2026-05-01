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
CANONICAL_HAND_STL_PATH = "/home/yiting/Documents/GitHub/hand_tracking/configs/Neo_Hand_Rigged_Canonical.stl"
CANONICAL_HAND_KPTS_PATH = "/home/yiting/Documents/GitHub/hand_tracking/configs/hand_keypoint_map.json"
WEIGHTS_JSON_PATH = "/home/yiting/Documents/GitHub/hand_tracking/configs/hand_skinning_weights.json" 

RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
STL_ROOT = Path("/media/yiting/NewVolume/Data/Shapes/shapes_stl")
OBJ_CONFIG_PATH = Path("/home/yiting/Documents/GitHub/hand_tracking/configs/obj_coordinates.json")

session_name = "2025-12-09"
trial_name = "2025-12-09_09-01-20"
FRAME_NUMBER = 300
DISTANCE_THRESHOLDS_MM = np.arange(-2, 4, 0.025)

# Hierarchy matching your Blender "Fan Rig"
finger_chains = {
    "Thumb": ["Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip"],
    "Index": ["Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip"],
    "Middle": ["Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip"],
    "Ring": ["Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip"],
    "Small": ["Small_MCP", "Small_PIP", "Small_DIP", "Small_Tip"]
}

# --- 1. LOAD CANONICAL DATA ---

canonical_hand_mesh = trimesh.load(CANONICAL_HAND_STL_PATH)
num_mesh_verts = len(canonical_hand_mesh.vertices)

with open(WEIGHTS_JSON_PATH, 'r') as f_in:
    weight_data = json.load(f_in)
num_weight_indices = len(weight_data)

print(f"Mesh Vertices: {num_mesh_verts}")
print(f"Weight Indices: {num_weight_indices}")

if num_mesh_verts != num_weight_indices:
    raise ValueError("CRITICAL ERROR: Mesh vertex count and weight count do not match!")

with open(CANONICAL_HAND_KPTS_PATH, "r") as f_in:
    canonical_hand_kpts = json.load(f_in)

def load_blender_weights(json_path, expected_vertices, bone_names):
    with open(json_path, 'r') as f_in:
        weight_data = json.load(f_in)
    
    num_in_json = len(weight_data)
    print(f"DEBUG: JSON contains {num_in_json} vertices. Expected {expected_vertices}.")
    
    if num_in_json != expected_vertices:
        raise ValueError(f"CRITICAL MISMATCH: Your weights file has {num_in_json} vertices, "
                         f"but your STL has {expected_vertices}. Please re-export BOTH from Blender.")

    bone_to_idx = {name: i for i, name in enumerate(bone_names)}
    weights_matrix = np.zeros((expected_vertices, len(bone_names)))
    
    for v_idx_str, influences in weight_data.items():
        v_idx = int(v_idx_str)
        # Check for out of bounds here before it causes a crash later
        if v_idx >= expected_vertices:
            continue 
            
        for b_name, val in influences.items():
            if b_name in bone_to_idx:
                weights_matrix[v_idx, bone_to_idx[b_name]] = val
    
    row_sums = weights_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    return weights_matrix / row_sums[:, np.newaxis]

def generate_bone_list(chains):
    bone_list = []
    # Metacarpals
    for name in chains.keys():
        bone_list.append(("Wrist_Root", chains[name][0], "palm", f"{name}_Metacarpal"))
    # Finger Segments
    for name, chain in chains.items():
        for i in range(len(chain)-1):
            bone_list.append((chain[i], chain[i+1], "finger", chain[i]))
    return bone_list

bones = generate_bone_list(finger_chains)
bone_names = [b[3] for b in bones]
weights = load_blender_weights(WEIGHTS_JSON_PATH, len(canonical_hand_mesh.vertices), bone_names)

# --- 2. LOAD TRIAL DATA ---
pose_3d_dir = ANALYSIS_ROOT / session_name / 'anipose' / 'pose_3d_filter'
df = pd.read_csv(pose_3d_dir / f'{trial_name}_f3d.csv')
target_frame_row = df.iloc[FRAME_NUMBER] # Renamed from 'f' to avoid confusion

def get_xyz(name, row):
    return np.array([row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"]])

# Load object metadata
log_path = RAW_DATA_ROOT / session_name / 'trial_logs' / f'{trial_name}_log.json'
with open(log_path, 'r') as f_log:
    log_data = json.load(f_log)
shape_id = log_data.get("shape_id", "unknown_0")
obj_id, orientation = shape_id.split("_")[0], shape_id.split("_")[-1]

obj_mesh = trimesh.load(STL_ROOT / f'{obj_id}.stl')
with open(OBJ_CONFIG_PATH, 'r') as f_obj:
    obj_configs = json.load(f_obj)

# Prepare trial keypoints and Wrist Root
trial_kpts = {k: get_xyz(k, target_frame_row) for k in canonical_hand_kpts.keys()}
trial_kpts["Wrist_Root"] = (trial_kpts["Wrist_R"] + trial_kpts["Wrist_U"]) / 2.0
canonical_hand_kpts["Wrist_Root"] = (np.array(canonical_hand_kpts["Wrist_R"]) + np.array(canonical_hand_kpts["Wrist_U"])) / 2.0

# --- 3. TRANSFORM FUNCTIONS ---

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

def pose_mesh_lbs_refined(mesh, weights, bone_list, c_kps, t_kps):
    num_vertices = len(mesh.vertices)
    num_weights = weights.shape[0]
    
    if num_vertices != num_weights:
        raise ValueError(f"Weights ({num_weights}) do not match Mesh ({num_vertices})!")

    bone_transforms = []
    
    def get_hand_normal(kp):
        v1 = kp["Middle_MCP"] - kp["Wrist_Root"]
        v2 = kp["Index_MCP"] - kp["Small_MCP"]
        n = np.cross(v1, v2)
        return n / np.linalg.norm(n)

    can_normal = get_hand_normal({k: np.array(v) for k, v in c_kps.items()})
    trial_normal = get_hand_normal(t_kps)

    for prox, dist, b_type, b_name in bone_list:
        cp, cd = np.array(c_kps[prox]), np.array(c_kps[dist])
        tp, td = t_kps[prox], t_kps[dist]

        v_can = (cd - cp) / np.linalg.norm(cd - cp)
        x_can = np.cross(v_can, can_normal); x_can /= np.linalg.norm(x_can)
        y_can = np.cross(x_can, v_can)
        M_src = np.eye(4); M_src[:3, :3] = np.column_stack([x_can, y_can, v_can]); M_src[:3, 3] = cp

        v_tri = (td - tp) / np.linalg.norm(td - tp)
        x_tri = np.cross(v_tri, trial_normal); x_tri /= np.linalg.norm(x_tri)
        y_tri = np.cross(x_tri, v_tri)
        M_tgt = np.eye(4); M_tgt[:3, :3] = np.column_stack([x_tri, y_tri, v_tri]); M_tgt[:3, 3] = tp

        bone_transforms.append(M_tgt @ np.linalg.inv(M_src))

    # Apply Skinning
    homog_v = np.hstack([mesh.vertices, np.ones((num_vertices, 1))])
    new_v = np.zeros((num_vertices, 4))
    
    for i in range(len(bone_list)):
        # Ensure we are applying transforms to all 8578 rows
        transformed = (bone_transforms[i] @ homog_v.T).T
        new_v += weights[:, i][:, np.newaxis] * transformed

    # CRITICAL: Use process=False to prevent trimesh from re-indexing your mesh!
    return trimesh.Trimesh(vertices=new_v[:, :3], faces=mesh.faces, process=False)

# --- 4. EXECUTION ---

# 1. Posed Reconstruction (8578 verts)
# Ensure your pose_mesh_lbs_refined uses process=False as discussed
posed_hand_mesh = pose_mesh_lbs_refined(canonical_hand_mesh, weights, bones, canonical_hand_kpts, trial_kpts)

# 2. Align Object STL to Trial Markers
obj_map = obj_configs["orientations"][orientation]
src_obj = np.array(list(obj_map.values()))
tgt_obj = np.array([get_xyz(name, target_frame_row) for name in obj_map.keys()])
R_obj, t_obj = get_rigid_transform(src_obj, tgt_obj)
obj_matrix = np.eye(4); obj_matrix[:3, :3] = R_obj; obj_matrix[:3, 3] = t_obj
obj_mesh.apply_transform(obj_matrix)

# 3. CONTACT SCORING
obj_tree = KDTree(obj_mesh.vertices)
# Query using the 8578 vertices of the posed hand
dists, _ = obj_tree.query(posed_hand_mesh.vertices)

# Build the score array (Size: 8578)
vertex_scores = np.zeros(len(posed_hand_mesh.vertices))
for t in DISTANCE_THRESHOLDS_MM:
    vertex_scores += (dists <= t).astype(int)

# --- 5. VISUALIZATION ---

def visualize_dual_view(posed_mesh, canonical_mesh, obj_mesh, scores, thresholds, trial_name):
    fig = plt.figure(figsize=(20, 10))
    norm = plt.Normalize(vmin=0, vmax=len(thresholds))
    
    # --- LEFT SUBPLOT: POSED INTERACTION ---
    ax1 = fig.add_subplot(121, projection='3d')
    # Use the 8578-sized scores array to color the faces
    face_colors_posed = cm.YlOrRd(norm(scores[posed_mesh.faces].mean(axis=1)))
    
    poly_posed = Poly3DCollection(posed_mesh.vertices[posed_mesh.faces], 
                                  facecolors=face_colors_posed, edgecolor='none', alpha=0.9, shade=True)
    ax1.add_collection3d(poly_posed)
    ax1.plot_trisurf(obj_mesh.vertices[:, 0], obj_mesh.vertices[:, 1], obj_mesh.vertices[:, 2], 
                     triangles=obj_mesh.faces, color='gray', alpha=0.2, edgecolor='none')

    # Scaling Logic
    all_v = np.vstack([posed_mesh.vertices, obj_mesh.vertices])
    mid = all_v.mean(axis=0); max_r = (all_v.max(axis=0) - all_v.min(axis=0)).max() / 2.0
    ax1.set_xlim(mid[0]-max_r, mid[0]+max_r); ax1.set_ylim(mid[1]-max_r, mid[1]+max_r); ax1.set_zlim(mid[2]-max_r, mid[2]+max_r)
    ax1.set_axis_off(); ax1.view_init(elev=20, azim=45)
    ax1.set_title(f"Posed Interaction: {trial_name}")

    # --- RIGHT SUBPLOT: CANONICAL HEATMAP ---
    ax2 = fig.add_subplot(122, projection='3d')
    # Map the same 8578 scores to the canonical faces (ventral view)
    face_colors_can = cm.YlOrRd(norm(scores[canonical_mesh.faces].mean(axis=1)))
    
    poly_can = Poly3DCollection(canonical_mesh.vertices[canonical_mesh.faces], 
                                facecolors=face_colors_can, edgecolor='none', alpha=1.0, shade=True)
    ax2.add_collection3d(poly_can)
    ax2.set_xlim(-50, 51); ax2.set_ylim(0, 60); ax2.set_zlim(-90, 30)
    ax2.set_axis_off(); ax2.view_init(elev=10, azim=-85)
    ax2.set_title("Standardized Canonical Heatmap")
    
    plt.show()

visualize_dual_view(posed_hand_mesh, canonical_hand_mesh, obj_mesh, vertex_scores, DISTANCE_THRESHOLDS_MM, trial_name)