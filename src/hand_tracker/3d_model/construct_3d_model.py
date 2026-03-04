import trimesh
import numpy as np
import open3d as o3d

# 1. Load your 3D-printed object
mesh = trimesh.load('your_object.stl')

# 2. Define your hand keypoints (Example data in mm)
# Replace this with your actual 3D tracing data for the single frame
hand_pts = np.array([
    [10.5, 20.1, 5.0], # Tip of Index
    [12.0, 18.5, 4.8], # Index PIP joint
    # ... add all 21 keypoints here
])

# 3. Calculate Distance from every vertex of the STL to the Hand
# 'flesh_radius' accounts for the distance from the bone to the skin surface
flesh_radius = 8.0  # mm (approximate for a human fingertip)

# Find the distance from every vertex on the STL to the nearest hand keypoint
distances = mesh.nearest.on_surface(hand_pts)[1] 

# 4. Identify Contact Areas
# We color the mesh based on proximity
# 0.0 distance = touching; > flesh_radius = not touching
colors = np.zeros((len(mesh.vertices), 4))
normalized_dist = np.clip(distances / flesh_radius, 0, 1)

# Color Map: Red for contact (close), White/Grey for far
colors[:, 0] = 1.0 - normalized_dist  # Red channel
colors[:, 1] = normalized_dist        # Green channel
colors[:, 2] = normalized_dist        # Blue channel
colors[:, 3] = 1.0                    # Alpha

mesh.visual.vertex_colors = colors

# 5. Export or Visualize
mesh.show()