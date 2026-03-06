import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import imageio
from PIL import Image

# --- CONFIGURATION ---
RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
STL_ROOT = Path("/media/yiting/NewVolume/Data/Shapes/shapes_stl")
session_name = "2025-12-09"
trial_name = "2025-12-09_09-02-01"
FRAME_NUMBER = 300

FINGER_DIAMETER_MM = 8.0  
PALM_THICKNESS = 9.0        
LW = (FINGER_DIAMETER_MM / 25.4) * 72 

OBJECT_COLOR = "#808080"    
HAND_COLORS = {
    "Thumb": "#FF5733", "Index": "#33FF57", "Middle": "#3357FF", 
    "Ring": "#F333FF", "Small": "#FFD433", "Palm": "#FFCC99"
}
HAND_OPACITY = 0.5  
OBJECT_OPACITY = 1 

ROTATE_XYZ = False  
SHIFT_ORIGIN = False  
NEW_REFERENCE = "Dot_t2"  

# --- 0. LOAD DATA ---
pose_3d_dir = ANALYSIS_ROOT / session_name / 'anipose' / 'pose_3d_filter'
csv_path = pose_3d_dir / f'{trial_name}_f3d.csv'
df = pd.read_csv(csv_path)
f = df.iloc[FRAME_NUMBER]

json_path = Path("/home/yiting/Documents/GitHub/hand_tracking/configs/obj_coordinates.json")
with open(json_path, 'r') as file:
    dot_configs = json.load(file)

log_path = RAW_DATA_ROOT / session_name / 'trial_logs' / f'{trial_name}_log.json'
with open(log_path, 'r') as file:
    log_data = json.load(file)
shape_id = log_data.get("shape_id", "unknown_0")
obj_id = shape_id.split("_")[0]
orientation = shape_id.split("_")[-1]

# Load STL
stl_path = STL_ROOT / f'{obj_id}.stl'
mesh = trimesh.load(stl_path)

# --- 1. FUNCTIONS ---
def get_point(name):
    """Extracts point, applies axes swap, and then 90 deg clockwise rotation."""
    p = np.array([f[f"{name}_x"], f[f"{name}_y"], f[f"{name}_z"]])
    if ROTATE_XYZ:
        # Step 1: Orbit X-axis swap [y, z, x]
        p_swap = np.array([p[1], p[2], p[0]])
        # Step 2: 90 Deg Clockwise [x'=y, y'=-x, z'=z]
        p_final = np.array([p_swap[1], -p_swap[0], p_swap[2]])
        return p_final
    return p

def rotate_mesh_vertices(vertices):
    """Applies the same rotation logic to mesh vertices."""
    if ROTATE_XYZ:
        v_swap = vertices[:, [1, 2, 0]]
        v_final = np.column_stack([v_swap[:, 1], -v_swap[:, 0], v_swap[:, 2]])
        return v_final
    return vertices

# --- 2. DEFINE HAND STRUCTURE ---
finger_chains = {
    "Small": ["Small_Tip", "Small_DIP", "Small_PIP", "Small_MCP"],
    "Ring": ["Ring_Tip", "Ring_DIP", "Ring_PIP", "Ring_MCP"],
    "Middle": ["Middle_Tip", "Middle_DIP", "Middle_PIP", "Middle_MCP"],
    "Index": ["Index_Tip", "Index_DIP", "Index_PIP", "Index_MCP"],
    "Thumb": ["Thumb_Tip", "Thumb_IP", "Thumb_MCP"]
}
palm_keypoints = ["Small_MCP", "Ring_MCP", "Middle_MCP", "Index_MCP", "Thumb_MCP", "Thumb_CMC", "Wrist_R", "Wrist_U"]

# --- 3. COORDINATE CALCULATIONS ---
mesh.vertices = rotate_mesh_vertices(mesh.vertices)

# Get object markers for alignment
dot_map = dot_configs["orientations"][orientation]
src_dots = np.array(list(dot_map.values()))
if ROTATE_XYZ:
    # Match the source dots to the new rotated world space
    src_swap = src_dots[:, [1, 2, 0]]
    src_dots = np.column_stack([src_swap[:, 1], -src_swap[:, 0], src_swap[:, 2]])

tgt_dots = np.array([get_point(name) for name in dot_map.keys()])

# Rigid Transform for STL alignment
def get_rigid_transform(A, B):
    centroid_A, centroid_B = np.mean(A, axis=0), np.mean(B, axis=0)
    H = (A - centroid_A).T @ (B - centroid_B)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    return R, centroid_B - R @ centroid_A

R, trans = get_rigid_transform(src_dots, tgt_dots)
mesh.vertices = (R @ mesh.vertices.T).T + trans

# Shifting Logic
new_origin = np.zeros(3)
if SHIFT_ORIGIN:
    new_origin = get_point(NEW_REFERENCE)
    mesh.vertices -= new_origin
    tgt_dots -= new_origin

# Palm Geometry (Hull) after shifts
palm_pts = np.array([get_point(name) for name in palm_keypoints]) - new_origin
wrist_avg = (get_point("Wrist_R") + get_point("Wrist_U")) / 2.0 - new_origin
v1 = (get_point("Middle_MCP") - new_origin) - wrist_avg
v2 = (get_point("Index_MCP") - new_origin) - (get_point("Small_MCP") - new_origin)
normal = np.cross(v1, v2)
normal /= np.linalg.norm(normal)

top_pts = palm_pts + (normal * (PALM_THICKNESS / 2.0))
bottom_pts = palm_pts - (normal * (PALM_THICKNESS / 2.0))
all_palm_cloud = np.vstack([top_pts, bottom_pts, palm_pts.mean(axis=0)])
hull = ConvexHull(all_palm_cloud)
hull_faces = [all_palm_cloud[s] for s in hull.simplices]

# --- 4. VISUALIZATION ---
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Fingers
for name, chain in finger_chains.items():
    pts = np.array([get_point(pt) for pt in chain]) - new_origin
    ax.plot(pts[:,0], pts[:,1], pts[:,2], color=HAND_COLORS[name], linewidth=LW, solid_capstyle='round', alpha=HAND_OPACITY, zorder=10)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='white', s=100, edgecolors='black', alpha=HAND_OPACITY, zorder=15)

# Palm
ax.add_collection3d(Poly3DCollection(hull_faces, facecolors=HAND_COLORS["Palm"], alpha=HAND_OPACITY, edgecolors='#555555', linewidths=0.1, zorder=1))
ax.scatter(palm_pts[:,0], palm_pts[:,1], palm_pts[:,2], color='white', s=100, edgecolors='black', alpha=HAND_OPACITY, zorder=15)

# Object
v, faces = mesh.vertices, mesh.faces
ax.plot_trisurf(v[:,0], v[:,1], v[:,2], triangles=faces, color=OBJECT_COLOR, alpha=OBJECT_OPACITY, edgecolor='none', zorder=1)
ax.scatter(tgt_dots[:,0], tgt_dots[:,1], tgt_dots[:,2], color='red', s=40, edgecolors='black', zorder=20)

ax.computed_zorder = False
ax.set_axis_off()

# Save PNG
recon_dir = ANALYSIS_ROOT / session_name / 'reconstructions'
recon_dir.mkdir(parents=True, exist_ok=True)
img_path = recon_dir / f'recon_{trial_name}_f{FRAME_NUMBER}.png'
plt.savefig(img_path, dpi=300)

# --- 5. GIF GENERATION ---
def create_rotation_gif(filename, fps=10):
    frames = []
    # Set fixed limits based on origin to prevent jumping
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-60, 60)

    for angle in range(0, 360, 10):
        ax.view_init(elev=20, azim=angle)
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(image)

    imageio.mimsave(filename, frames, fps=fps, loop=0)
    print(f"GIF saved to {filename}")

gif_path = img_path.with_suffix('.gif')
create_rotation_gif(gif_path)