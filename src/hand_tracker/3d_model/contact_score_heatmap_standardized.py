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

FINGER_DIAMETER_MM = 8  # Approximate diameter of a monkey finger in mm for visualization purposes
PALM_THICKNESS = 9        # Palm thickness in mm
LW = (FINGER_DIAMETER_MM / 25.4) * 72 # 0.8mm in points (1 inch = 25.4mm, 1 inch = 72 points)

HAND_COLOR = "#FFCC99"
HAND_OPACITY = 0.1 
OBJECT_OPACITY = 0.5 
DISTANCE_THRESHOLDS_MM = np.arange(-3, 6, 0.01)  # Thresholds for contact depth in mm

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


# --- 3. Get Hand Surface Points    ---
def get_finger_surface_points(finger_chains, normal, finger_diameter=FINGER_DIAMETER_MM):
    """
    Finger sampling:
    1. Spheres at every joint (MCP, PIP, DIP, Tip) to seal gaps and create pads.
    2. Dense cylinders for bone segments.
    3. All points filtered by the palm normal to keep only object-facing surfaces.
    """
        
    surface_points = []
    radius = finger_diameter / 2.0
    

    for name, chain in finger_chains.items():
        # --- 1. SAMPLE JOINT SPHERES (including Tips) ---
        for joint_name in chain:
            joint_center = get_xyz(joint_name)
            
            # Use higher density for Tip spheres if desired
            res = 25 if "Tip" in joint_name else 20
            
            # Generate sphere points
            for phi in np.linspace(0, np.pi, res):
                for theta in np.linspace(0, 2*np.pi, res):
                    # Sphere coordinates
                    dx = radius * np.sin(phi) * np.cos(theta)
                    dy = radius * np.sin(phi) * np.sin(theta)
                    dz = radius * np.cos(phi)
                    
                    test_pt = joint_center + np.array([dx, dy, dz])
                    
                    # Keep points facing the object (heuristic: dot with -normal)
                    if np.dot(test_pt - joint_center, -normal) > 0:
                        surface_points.append(test_pt)

        # --- 2. SAMPLE BONE CYLINDERS ---
        for i in range(len(chain) - 1):
            p1 = get_xyz(chain[i])     # Distal joint
            p2 = get_xyz(chain[i+1])   # Proximal joint
            
            v_seg = p2 - p1
            dist_seg = np.linalg.norm(v_seg)
            z_axis = v_seg / dist_seg
            
            # Create a local frame for the cylinder cross-section
            ref_vec = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            x_axis = np.cross(ref_vec, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Density: 0.5mm longitudinal steps, 24 radial points
            num_steps = int(dist_seg / 0.5)
            num_angles = 24
            
            for s in np.linspace(0, 1, num_steps):
                center = p1 + s * v_seg
                for angle in np.linspace(0, 2 * np.pi, num_angles):
                    test_pt = center + radius * (np.cos(angle) * x_axis + np.sin(angle) * y_axis)
                    
                    # Keep points facing the object
                    if np.dot(test_pt - center, -normal) > 0:
                        surface_points.append(test_pt)

    return np.array(surface_points)


def sample_palm_surface_points(all_palm_cloud, hull, normal, num_samples=10000):
    """
    Samples points from the hull and filters for the side facing the object.
    """
    simplices = hull.simplices
    
    # Calculate areas for area-weighted sampling
    def tri_area(p1, p2, p3):
        return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

    areas = np.array([tri_area(*all_palm_cloud[s]) for s in simplices])
    probs = areas / np.sum(areas)
    
    # Sample triangles
    chosen_indices = np.random.choice(len(simplices), size=num_samples, p=probs)
    
    sampled_points = []
    palm_center = all_palm_cloud.mean(axis=0)

    for idx in chosen_indices:
        tri_pts = all_palm_cloud[simplices[idx]]
        # Barycentric sampling
        r1, r2 = np.sqrt(np.random.random()), np.random.random()
        pt = (1 - r1) * tri_pts[0] + r1 * (1 - r2) * tri_pts[1] + r1 * r2 * tri_pts[2]
        
        # FILTER: Keep only points on the "bottom" (palm) side
        # Vector from center to point should be in direction of -normal
        if np.dot(pt - palm_center, -normal) > 0:
            sampled_points.append(pt)
            
    return np.array(sampled_points)


# Calculate Normal to find Palm 'down' direction
wrist_avg = (get_xyz("Wrist_R") + get_xyz("Wrist_U")) / 2.0
v1 = get_xyz("Middle_MCP") - wrist_avg
v2 = get_xyz("Index_MCP") - get_xyz("Small_MCP")
normal = np.cross(v1, v2)
normal /= np.linalg.norm(normal)

# Get Finger Surface Points
finger_surface_points = get_finger_surface_points(finger_chains, normal)

# Get Palm Surface Points
# Generate Palm Volume points (shifted +4.5/-4.5mm)
top_pts = palm_pts + (normal * (PALM_THICKNESS / 2.0))
bottom_pts = palm_pts - (normal * (PALM_THICKNESS / 2.0))
all_palm_cloud = np.vstack([top_pts, bottom_pts, palm_pts.mean(axis=0)])
hull = ConvexHull(all_palm_cloud)
hull_faces = [all_palm_cloud[s] for s in hull.simplices]

palm_surface_points = sample_palm_surface_points(all_palm_cloud, hull, normal)

hand_surface_points = np.vstack([finger_surface_points, palm_surface_points])

# --- 4. CUMULATIVE CONTACT SCORE CALCULATION ---
def get_contact_scores(surface_points, obj_tree, thresholds):
    """
    Assigns a score to each point based on how many distance thresholds it passes.
    """
    distances, _ = obj_tree.query(surface_points)
    
    # Initialize scores (Points start at 0)
    scores = np.zeros(len(surface_points))
    
    # A point gets +1 for every threshold it is "deeper" than
    # e.g., if dist is -2mm, it satisfies thresholds of [5, 2, 0, -2]
    for t in thresholds:
        scores += (distances <= t).astype(int)
        
    return scores, distances

# Calculate scores
scores, dists = get_contact_scores(hand_surface_points, obj_tree, DISTANCE_THRESHOLDS_MM)

# --- 5. VISUALIZATION (Heatmap) ---
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# A. Plot Fingers (Heatmapped Cylinders)
# for name, chain in finger_chains.items():
#     pts = np.array([get_xyz(pt) for pt in chain])
#     ax.plot(pts[:,0], pts[:,1], pts[:,2], color=HAND_COLOR, 
#             linewidth=LW, solid_capstyle='round', alpha=HAND_OPACITY, zorder=10)
    # Add joints as white spheres with black edges
    # ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='white', s=70, edgecolors='black', zorder=15)

# B. Plot Palm Volume (Heatmapped Convex Hull)
palm_volume = Poly3DCollection(hull_faces, facecolors=HAND_COLOR, 
                               edgecolors='k', linewidths=0.2, alpha=HAND_OPACITY, zorder=5)
# ax.add_collection3d(palm_volume)

# C. Plot the STL Object (Semi-Transparent Grey)
ax.plot_trisurf(v_mesh[:,0], v_mesh[:,1], v_mesh[:,2], 
                triangles=faces_mesh, color='gray', alpha=OBJECT_OPACITY, 
                edgecolor='none', zorder=1)

# E. Map scores to a color gradient
norm = plt.Normalize(vmin=0, vmax=len(DISTANCE_THRESHOLDS_MM))

ax.scatter(hand_surface_points[:,0], hand_surface_points[:,1], hand_surface_points[:,2], 
           c=scores, cmap='YlOrRd', s=2, alpha=0.9, edgecolors='none')


# --- 6. FORMATTING ---
ax.set_title("3D Hand Reconstruction: Contact Score Heatmap")
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
img_path = recon_dir / f'contact_scores_{trial_name}_f{FRAME_NUMBER}.png'
plt.savefig(img_path, dpi=300)


# def generate_standardized_hand_map(finger_chains, hand_surface, scores, resolution=128):
#     """
#     Creates a standardized 2D heatmap. 
#     Top 60%: Fingers (Thumb to Small)
#     Bottom 40%: Palm Quad projection
#     """
#     flat_map = np.zeros((resolution, resolution))
    
#     # 1. Split definitions
#     v_split = int(resolution * 0.4)
#     finger_names = ["Thumb", "Index", "Middle", "Ring", "Small"]
#     col_width = resolution // len(finger_names)

#     # --- A. FINGER MAPPING (Vertical range: [v_split, resolution]) ---
#     for f_idx, name in enumerate(finger_names):
#         chain = finger_chains[name]
#         joints = [get_xyz(j) for j in chain]
#         bone_lengths = [np.linalg.norm(joints[i] - joints[i+1]) for i in range(len(joints)-1)]
#         total_skeletal_len = sum(bone_lengths)
        
#         # Determine u-range for this finger column
#         u_start, u_end = f_idx * col_width, (f_idx + 1) * col_width

#         for pt, score in zip(hand_surface, scores):
#             for i in range(len(joints)-1):
#                 p_distal, p_prox = joints[i], joints[i+1]
#                 bone_vec = p_distal - p_prox
#                 bone_unit = bone_vec / np.linalg.norm(bone_vec)
                
#                 proj = np.dot(pt - p_prox, bone_unit)
#                 dist_to_axis = np.linalg.norm((pt - p_prox) - proj * bone_unit)
                
#                 # Check if point belongs to this bone segment (6mm radius threshold)
#                 if 0 <= proj <= np.linalg.norm(bone_vec) and dist_to_axis < 6.0:
#                     # Normalized position from MCP (0.0) to Tip (1.0)
#                     len_from_mcp = sum(bone_lengths[k] for k in range(i+1, len(joints)-1)) + proj
#                     norm_v = len_from_mcp / total_skeletal_len
                    
#                     v_idx = v_split + int(norm_v * (resolution - v_split - 1))
#                     # Use max pooling to keep the highest contact score for this region
#                     flat_map[v_idx, u_start:u_end] = np.maximum(flat_map[v_idx, u_start:u_end], score)

#     # --- B. PALM MAPPING (Vertical range: [0, v_split]) ---
#     p_idx_mcp = get_xyz("Index_MCP")
#     p_sml_mcp = get_xyz("Small_MCP")
#     p_wri_r   = get_xyz("Wrist_R")
#     p_wri_u   = get_xyz("Wrist_U")

#     # Defined axes for the palm quad
#     v_axis = ((p_idx_mcp + p_sml_mcp)/2) - ((p_wri_r + p_wri_u)/2)
#     u_axis = p_sml_mcp - p_idx_mcp

#     for pt, score in zip(hand_surface, scores):
#         # Heuristic: Check if point is closer to palm center than tips
#         palm_center = (p_idx_mcp + p_sml_mcp + p_wri_r + p_wri_u) / 4.0
#         if np.linalg.norm(pt - palm_center) < 25.0:
#             # Project onto local U (radial-ulnar) and V (wrist-mcp) axes
#             u_palm = np.dot(pt - p_idx_mcp, u_axis) / np.dot(u_axis, u_axis)
#             v_palm = np.dot(pt - p_wri_r, v_axis) / np.dot(v_axis, v_axis)
            
#             if 0 <= u_palm <= 1 and 0 <= v_palm <= 1:
#                 u_idx = int(u_palm * (resolution - 1))
#                 v_idx = int(v_palm * (v_split - 1))
#                 flat_map[v_idx, u_idx] = max(flat_map[v_idx, u_idx], score)

#     return flat_map

# # --- 2. VISUALIZATION ---
# standardized_heatmap = generate_standardized_hand_map(finger_chains, hand_surface_points, scores)

# plt.figure(figsize=(6, 8))
# plt.imshow(standardized_heatmap, cmap='YlOrRd', origin='lower', aspect='auto')
# plt.axhline(y=int(128*0.4), color='black', linestyle='--', label='MCP Line')
# plt.title("Standardized Hand Contact Map")
# plt.xlabel("Fingers (Thumb $\\rightarrow$ Small)")
# plt.ylabel("Proximal $\\rightarrow$ Distal")
# plt.colorbar(label='Contact Score')
# # plt.show()

# # Save PNG
# recon_dir = ANALYSIS_ROOT / session_name / 'reconstructions' / trial_name
# recon_dir.mkdir(parents=True, exist_ok=True)
# img_path = recon_dir / f'contact_scores_standardized_heatmap_{trial_name}_f{FRAME_NUMBER}.png'
# plt.savefig(img_path, dpi=300)