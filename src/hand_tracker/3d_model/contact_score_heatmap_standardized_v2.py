import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, KDTree

# --- CONFIGURATION ---
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
STL_ROOT = Path("/media/yiting/NewVolume/Data/Shapes/shapes_stl")
session_name = "2025-12-09"
trial_name = "2025-12-09_09-01-20"
FRAME_NUMBER = 300

FINGER_DIAMETER_MM = 8.0  
PALM_THICKNESS = 9.0        
DISTANCE_THRESHOLDS_MM = np.arange(-5, 10, 0.5)

# --- 1. DATA LOADING & OBJECT ALIGNMENT ---

# Load 3D pose data
pose_3d_dir = ANALYSIS_ROOT / session_name / 'anipose' / 'pose_3d_filter'
df = pd.read_csv(pose_3d_dir / f'{trial_name}_f3d.csv')
f = df.iloc[FRAME_NUMBER] # The 'f' variable

# Load object and orientation info
log_path = RAW_DATA_ROOT / session_name / 'trial_logs' / f'{trial_name}_log.json'
with open(log_path, 'r') as file:
    log_data = json.load(file)
shape_id = log_data.get("shape_id", "unknown_0")
obj_id, orientation = shape_id.split("_")[0], shape_id.split("_")[-1]

# Load STL and Configs
mesh = trimesh.load(STL_ROOT / f'{obj_id}.stl')
with open(Path("/home/yiting/Documents/GitHub/hand_tracking/configs/obj_coordinates.json"), 'r') as file:
    dot_configs = json.load(file)

def get_xyz(name, frame_data):
    return np.array([frame_data[f"{name}_x"], frame_data[f"{name}_y"], frame_data[f"{name}_z"]])

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

# Align STL to tracked markers
dot_map = dot_configs["orientations"][orientation]
src_dots = np.array(list(dot_map.values()))
tgt_dots = np.array([get_xyz(name, f) for name in dot_map.keys()])
R, t = get_rigid_transform(src_dots, tgt_dots)

matrix = np.eye(4)
matrix[:3, :3] = R
matrix[:3, 3] = t
mesh.apply_transform(matrix)

# Initialize 'obj_tree' for distance queries
obj_tree = KDTree(mesh.vertices)

# --- 2. CANONICAL & POSED MESH LOGIC ---

finger_chains = {
    "Small": ["Small_Tip", "Small_DIP", "Small_PIP", "Small_MCP"],
    "Ring": ["Ring_Tip", "Ring_DIP", "Ring_PIP", "Ring_MCP"],
    "Middle": ["Middle_Tip", "Middle_DIP", "Middle_PIP", "Middle_MCP"],
    "Index": ["Index_Tip", "Index_DIP", "Index_PIP", "Index_MCP"],
    "Thumb": ["Thumb_Tip", "Thumb_IP", "Thumb_MCP"]
}
palm_loop = ["Small_MCP", "Ring_MCP", "Middle_MCP", "Index_MCP", "Thumb_MCP", "Thumb_CMC", "Wrist_R", "Wrist_U"]

def get_canonical_flat_mesh(finger_chains, palm_loop):
    parts = []
    radius = FINGER_DIAMETER_MM / 2.0
    spacing = 16.0
    canonical_joints = {
        "Wrist_R": np.array([-10, -30, 0]), "Wrist_U": np.array([10, -30, 0]),
        "Thumb_CMC": np.array([-25, -15, 0]), "Thumb_MCP": np.array([-30, 5, 0]),
        "Thumb_IP":  np.array([-35, 20, 0]),  "Thumb_Tip": np.array([-38, 32, 0])
    }
    for i, name in enumerate(["Thumb", "Index", "Middle", "Ring", "Small"]):
        if name == "Thumb": continue
        x_off = (i - 2) * spacing
        canonical_joints[f"{name}_MCP"] = np.array([x_off, 0, 0])
        canonical_joints[f"{name}_PIP"] = np.array([x_off, 18, 0])
        canonical_joints[f"{name}_DIP"] = np.array([x_off, 33, 0])
        canonical_joints[f"{name}_Tip"] = np.array([x_off, 45, 0])

    for name, chain in finger_chains.items():
        for i, joint_name in enumerate(chain):
            p = canonical_joints[joint_name]
            parts.append(trimesh.creation.uv_sphere(radius=radius, count=[12, 12]).apply_translation(p))
            if i < len(chain) - 1:
                p_prox = canonical_joints[chain[i+1]]
                bone_vec = p - p_prox
                cyl = trimesh.creation.cylinder(radius=radius, height=np.linalg.norm(bone_vec), sections=12)
                cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], bone_vec))
                cyl.apply_translation((p + p_prox) / 2.0)
                parts.append(cyl)

    palm_pts_flat = np.array([canonical_joints[k] for k in palm_loop])
    flat_palm_cloud = np.vstack([palm_pts_flat + [0,0,4.5], palm_pts_flat - [0,0,4.5], [0,-15,0]])
    parts.append(trimesh.Trimesh(vertices=flat_palm_cloud, faces=ConvexHull(flat_palm_cloud).simplices))
    return trimesh.util.concatenate(parts)

def calculate_trial_mesh_scores(f, finger_chains, palm_loop, obj_tree):
    parts = []
    radius = FINGER_DIAMETER_MM / 2.0
    for name, chain in finger_chains.items():
        for i, joint_name in enumerate(chain):
            p = get_xyz(joint_name, f)
            parts.append(trimesh.creation.uv_sphere(radius=radius, count=[12, 12]).apply_translation(p))
            if i < len(chain) - 1:
                p_prox = get_xyz(chain[i+1], f)
                bone_vec = p - p_prox
                cyl = trimesh.creation.cylinder(radius=radius, height=np.linalg.norm(bone_vec), sections=12)
                cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], bone_vec))
                cyl.apply_translation((p + p_prox) / 2.0)
                parts.append(cyl)

    p_palm = np.array([get_xyz(k, f) for k in palm_loop])
    v1 = get_xyz("Middle_MCP", f) - ((get_xyz("Wrist_R", f)+get_xyz("Wrist_U", f))/2)
    v2 = get_xyz("Index_MCP", f)-get_xyz("Small_MCP", f)
    norm = np.cross(v1, v2); norm /= np.linalg.norm(norm)
    posed_palm_cloud = np.vstack([p_palm + (norm*4.5), p_palm - (norm*4.5), p_palm.mean(axis=0)])
    parts.append(trimesh.Trimesh(vertices=posed_palm_cloud, faces=ConvexHull(posed_palm_cloud).simplices))

    full_posed = trimesh.util.concatenate(parts)
    dists, _ = obj_tree.query(full_posed.vertices)
    return np.sum([dists <= t for t in DISTANCE_THRESHOLDS_MM], axis=0)

# --- 3. EXECUTION & VISUALIZATION ---

vertex_scores = calculate_trial_mesh_scores(f, finger_chains, palm_loop, obj_tree)
flat_canvas = get_canonical_flat_mesh(finger_chains, palm_loop)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Map trial scores to the static flat canvas faces
norm = plt.Normalize(vmin=0, vmax=len(DISTANCE_THRESHOLDS_MM))
face_colors = cm.YlOrRd(norm(vertex_scores[flat_canvas.faces].mean(axis=1)))

poly = Poly3DCollection(flat_canvas.vertices[flat_canvas.faces], facecolors=face_colors, edgecolor='none', shade=True)
ax.add_collection3d(poly)

ax.view_init(elev=90, azim=-90)
ax.set_xlim(-50, 50); ax.set_ylim(-40, 60); ax.set_zlim(-10, 10)
ax.set_axis_off()
plt.title(f"Canonical Contact Pattern: {trial_name}")
plt.show()