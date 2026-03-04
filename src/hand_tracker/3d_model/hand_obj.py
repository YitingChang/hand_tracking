import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
STL_ROOT = Path("/media/yiting/NewVolume/Data/Shapes/shapes_stl")
session_name = "2025-12-09"
trial_name = "2025-12-09_09-02-01"
FRAME_NUMBER = 300
# Constants for thickness
FINGER_DIAMETER_MM = 8.0  # Approximate diameter of a monkey finger in mm for visualization purposes
PALM_THICKNESS = 9.0        # Palm thickness in mm
# Conversion to Matplotlib points
LW = (FINGER_DIAMETER_MM / 25.4) * 72 # 0.8mm in points (1 inch = 25.4mm, 1 inch = 72 points)
OBJECT_COLOR = "#808080"    # Neutral grey for the object
HAND_COLORS = {
    "Thumb": "#FF5733",  # Orange-Red
    "Index": "#33FF57",  # Green
    "Middle": "#3357FF", # Blue
    "Ring": "#F333FF",   # Magenta
    "Small": "#FFD433",  # Yellow
    "Palm": "#FFCC99"    # Realistic skin-tone/peach for the palm
}

# --- 1. LOAD DATA ---
# Load 3D pose data from CSV
pose_3d_dir = ANALYSIS_ROOT / session_name / 'anipose' / 'pose_3d_filter'
csv_file = f'{trial_name}_f3d.csv'
csv_path = pose_3d_dir / csv_file
df = pd.read_csv(csv_path)
f = df.iloc[FRAME_NUMBER]

# Load object dot positions in a stimulus coordinate system 
json_path = Path("/home/yiting/Documents/GitHub/hand_tracking/configs/obj_coordinates.json")
with open(json_path, 'r') as file:
    dot_configs = json.load(file)

# Load logging info for trial (to get stimulus orientation)
log_path = RAW_DATA_ROOT / session_name / 'trial_logs' / f'{trial_name}_log.json'
with open(log_path, 'r') as file:
    log_data = json.load(file)
shape_id = log_data.get("shape_id", "unknown_0")
obj_id = shape_id.split("_")[0]  # Assuming shape_id format includes object ID at the start
orientation = shape_id.split("_")[-1]  # Assuming shape_id format includes orientation at the end

# Load STL mesh for the object
stl_path = STL_ROOT / f'{obj_id}.stl'
mesh = trimesh.load(stl_path)

def get_xyz(name):
    """Helper to extract XYZ from CSV for a given keypoint name."""
    return np.array([f[f"{name}_x"], f[f"{name}_y"], f[f"{name}_z"]])

# --- 2. DEFINE HAND STRUCTURE ---
# Define the connection chains
finger_chains = {
    "Small": ["Small_Tip", "Small_DIP", "Small_PIP", "Small_MCP"],
    "Ring": ["Ring_Tip", "Ring_DIP", "Ring_PIP", "Ring_MCP"],
    "Middle": ["Middle_Tip", "Middle_DIP", "Middle_PIP", "Middle_MCP"],
    "Index": ["Index_Tip", "Index_DIP", "Index_PIP", "Index_MCP"],
    "Thumb": ["Thumb_Tip", "Thumb_IP", "Thumb_MCP"]
}

# --- 3. PALM GEOMETRY (Convex Hull) ---
palm_keypoints = ["Small_MCP", "Ring_MCP", "Middle_MCP", "Index_MCP", 
                  "Thumb_MCP", "Thumb_CMC", "Wrist_R", "Wrist_U"]
base_pts = np.array([get_xyz(name) for name in palm_keypoints])

# Define the local normal to create the volume
wrist_avg = (get_xyz("Wrist_R") + get_xyz("Wrist_U")) / 2.0
v1 = get_xyz("Middle_MCP") - wrist_avg
v2 = get_xyz("Index_MCP") - get_xyz("Small_MCP")
normal = np.cross(v1, v2)
normal /= np.linalg.norm(normal)

# Generate a cloud of points for the palm volume
# Including the center point makes the hull more "fleshy"
palm_center = base_pts.mean(axis=0)
top_pts = base_pts + (normal * (PALM_THICKNESS / 2.0))
bottom_pts = base_pts - (normal * (PALM_THICKNESS / 2.0))
all_palm_cloud = np.vstack([top_pts, bottom_pts, palm_center])

# Calculate the Convex Hull
hull = ConvexHull(all_palm_cloud)
hull_faces = [all_palm_cloud[s] for s in hull.simplices]

# --- 4. CALCULATE RIGID TRANSFORM FOR STL ---
def get_rigid_transform(A, B):
    """Calculates R and t such that B = R*A + t (Procrustes Analysis)"""
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

# Get source dots from JSON config based on orientation
dot_map = dot_configs["orientations"][orientation]
src_dots = np.array(list(dot_map.values()))

# Get target dots from CSV tracking
dot_names = list(dot_map.keys())
tgt_dots = np.array([get_xyz(name) for name in dot_names])

# Calculate transform and apply to STL mesh
R, t = get_rigid_transform(src_dots, tgt_dots)
matrix = np.eye(4)
matrix[:3, :3] = R
matrix[:3, 3] = t
mesh.apply_transform(matrix)

# --- 5. VISUALIZATION ---
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# A. Plot Fingers (8mm Cylinders)
for name, chain in finger_chains.items():
    pts = np.array([get_xyz(pt) for pt in chain])
    ax.plot(pts[:,0], pts[:,1], pts[:,2], color=HAND_COLORS[name], 
            linewidth=LW, solid_capstyle='round', alpha=0.5, zorder=10)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='white', s=40, edgecolors='black', zorder=15)

# B. Plot Realistic Palm (Convex Hull)
palm_poly = Poly3DCollection(hull_faces, facecolors=HAND_COLORS["Palm"], alpha=0.5, zorder=5,
                             edgecolors='k', linewidths=0.2)
ax.add_collection3d(palm_poly)
ax.scatter(base_pts[:,0], base_pts[:,1], base_pts[:,2], 
           color='white', s=40, edgecolors='black', zorder=13)

# C. Plot Object (Solid Grey Surface)
# mesh = trimesh.load('transformed_object_f300.stl') # Assuming object is pre-aligned
v, faces = mesh.vertices, mesh.faces
ax.plot_trisurf(v[:,0], v[:,1], v[:,2], triangles=faces, 
                color=OBJECT_COLOR, alpha=0.6, edgecolor='none')

# D. Plot Markers (Red)
dot_names = ["Dot_t1", "Dot_t2", "Dot_t3", "Dot_b1", "Dot_b2", "Dot_b3", 
             "Dot_l1", "Dot_l2", "Dot_l3", "Dot_r1", "Dot_r2", "Dot_r3"]
tgt_dots = np.array([get_xyz(name) for name in dot_names])
ax.scatter(tgt_dots[:,0], tgt_dots[:,1], tgt_dots[:,2], 
           color='red', s=40, edgecolors='black', label='Object Markers', zorder=20)

# --- 4. FINAL FORMATTING ---
ax.set_title(f"Hand and Object (Frame {FRAME_NUMBER}, Shape {shape_id})", fontsize=16)
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')

# Set Equal Aspect Ratio
all_geometry = np.vstack([v, all_palm_cloud])
max_range = (all_geometry.max(axis=0) - all_geometry.min(axis=0)).max() / 2.0
mid = all_geometry.mean(axis=0)
ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

plt.show()
