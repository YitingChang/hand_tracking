from pathlib import Path
import json
import pandas as pd
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, KDTree

RAW_DATA_ROOT = Path("/media/yiting/NewVolume/Data/Videos")
ANALYSIS_ROOT = Path("/media/yiting/NewVolume/Analysis")
STL_ROOT = Path("/media/yiting/NewVolume/Data/Shapes/shapes_stl")
session_name = "2025-12-09"
trial_name = "2025-12-09_09-01-20"
FRAME_NUMBER = 300

FINGER_DIAMETER_MM = 12  # Approximate diameter of a monkey finger in mm for visualization purposes
PALM_THICKNESS = 13        # Palm thickness in mm
LW = (FINGER_DIAMETER_MM / 25.4) * 72 # 0.8mm in points (1 inch = 25.4mm, 1 inch = 72 points)

# Heatmap Color Gradient (Muted Beige -> Red)
# You can change 'OrRd' to 'RdYlGn_r' for (Green -> Red)
COLOR_MAP = plt.colormaps.get_cmap('YlOrRd')
NO_CONTACT_COLOR = "#D3D3D3" # Light Gray/Beige for far away

HAND_OPACITY = 0.5 
OBJECT_OPACITY = 1 

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
    return np.array([f[f"{name}_x"], f[f"{name}_y"], f[f"{name}_z"]])

# Define connection chains
finger_chains = {
    "Small": ["Small_Tip", "Small_DIP", "Small_PIP", "Small_MCP"],
    "Ring": ["Ring_Tip", "Ring_DIP", "Ring_PIP", "Ring_MCP"],
    "Middle": ["Middle_Tip", "Middle_DIP", "Middle_PIP", "Middle_MCP"],
    "Index": ["Index_Tip", "Index_DIP", "Index_PIP", "Index_MCP"],
    "Thumb": ["Thumb_Tip", "Thumb_IP", "Thumb_MCP"]
}
palm_loop = ["Small_MCP", "Ring_MCP", "Middle_MCP", "Index_MCP", "Thumb_MCP", "Thumb_CMC", "Wrist_R", "Wrist_U"]
palm_pts = np.array([get_xyz(pt) for pt in palm_loop])

# --- 2. CALCULATE RIGID TRANSFORM FOR STL ---
def get_rigid_transform(src, tgt):
    """Calculates R and t such that B = R*A + t (Procrustes Analysis)"""
    mask = ~np.isnan(src).any(axis=1) & ~np.isnan(tgt).any(axis=1)
    A = src[mask]
    B = tgt[mask]
    
    if len(A) < 3:
        raise ValueError("Not enough valid markers to calculate transform.")
    
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

# Get mesh vertices and faces for distance calculations
v_mesh, faces_mesh = mesh.vertices, mesh.faces
obj_tree = KDTree(v_mesh) # KDTree for fast distance lookup

# --- 3. HEATMAP LOGIC ---
def get_contact_color(points, threshold_max=15.0):
    """
    Calculates the contact color for a set of points based on 
    minimum distance to the STL object.
    
    Returns: A hex color string.
    """
    # 1. Find min distance from the group of points to the object
    # For a segment, we check its midpoint or joints
    distances, _ = obj_tree.query(points)
    min_dist = np.min(distances)
    
    # 2. Map distance (0mm to threshold_max) to a probability (1.0 to 0.0)
    # 0mm = Red (1.0), 15mm+ = Gray/Beige (0.0)
    contact_prob = max(0.0, min(1.0, 1.0 - (min_dist / threshold_max)))
    
    # 3. Handle 'No Contact' case (far away)
    if contact_prob < 0.05:
        return NO_CONTACT_COLOR
        
    # 4. Blend the color using the colormap
    rgba = COLOR_MAP(contact_prob)
    return colors.to_hex(rgba)


# --- 4. PLOTTING ---
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# A. Plot Fingers (Heatmapped Cylinders)
# Instead of plotting the whole chain at once, we plot segment-by-segment 
# to get better color resolution along the finger
for chain_name, chain in finger_chains.items():
    for i in range(len(chain) - 1):
        p1 = get_xyz(chain[i])
        p2 = get_xyz(chain[i+1])
        segment_pts = np.vstack([p1, p2])
        
        # Calculate color based on segment's proximity to object
        # Using a tighter threshold for fingers (10mm)
        seg_color = get_contact_color(segment_pts, threshold_max=10.0)
        
        ax.plot(segment_pts[:,0], segment_pts[:,1], segment_pts[:,2], 
                color=seg_color, linewidth=LW, 
                solid_capstyle='round', alpha=HAND_OPACITY, zorder=10)
        
        # # Plot Joint Markers (White Spheres)
        # ax.scatter(p1[0], p1[1], p1[2], color='white', s=50, 
        #            edgecolors='black', zorder=15)

# B. Plot Palm Volume (Heatmapped Convex Hull)
# Calculate Normal to find Palm 'down' direction
wrist_avg = (get_xyz("Wrist_R") + get_xyz("Wrist_U")) / 2.0
v1 = get_xyz("Middle_MCP") - wrist_avg
v2 = get_xyz("Index_MCP") - get_xyz("Small_MCP")
normal = np.cross(v1, v2)
normal /= np.linalg.norm(normal)

# Generate Palm Volume points (shifted +4.5/-4.5mm)
top_pts = palm_pts + (normal * (PALM_THICKNESS / 2.0))
bottom_pts = palm_pts - (normal * (PALM_THICKNESS / 2.0))
all_palm_cloud = np.vstack([top_pts, bottom_pts, palm_pts.mean(axis=0)])

# Calculate the Convex Hull & Its Heatmap Color
# Since the palm is big, we check the bottom faces specifically
palm_color = get_contact_color(bottom_pts, threshold_max=PALM_THICKNESS + 5.0)

hull = ConvexHull(all_palm_cloud)
hull_faces = [all_palm_cloud[s] for s in hull.simplices]

palm_volume = Poly3DCollection(hull_faces, facecolors=palm_color, 
                               edgecolors='k', linewidths=0.2, alpha=HAND_OPACITY, zorder=5)
ax.add_collection3d(palm_volume)

# C. Plot the STL Object (Semi-Transparent Grey)
ax.plot_trisurf(v_mesh[:,0], v_mesh[:,1], v_mesh[:,2], 
                triangles=faces_mesh, color='gray', alpha=OBJECT_OPACITY, 
                edgecolor='none', zorder=1)

# D. Plot the 12 Object Markers (Red Dots)
# tgt_dots = np.array([get_xyz(name) for name in dot_names]) # (dot_names from CSV)
# ax.scatter(tgt_dots[:,0], tgt_dots[:,1], tgt_dots[:,2], 
#            color='red', s=20, edgecolors='black', label='Object Markers', zorder=20)

# --- 4. FORMATTING ---
ax.set_title("3D Hand Reconstruction: Contact Heatmap")
ax.set_axis_off() # Cleaner GIF view

# Equal Aspect Ratio
all_pts = np.vstack([v_mesh, top_pts, bottom_pts])
max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
mid = all_pts.mean(axis=0)
ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

ax.computed_zorder = False
ax.view_init(elev=3, azim=82, roll=-10)
# plt.show()

# Save PNG
recon_dir = ANALYSIS_ROOT / session_name / 'reconstructions' / trial_name
recon_dir.mkdir(parents=True, exist_ok=True)
img_path = recon_dir / f'contact_heatmap_{trial_name}_f{FRAME_NUMBER}_t{FINGER_DIAMETER_MM}.png'
plt.savefig(img_path, dpi=300)