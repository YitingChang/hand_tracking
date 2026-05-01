import trimesh
import json
import numpy as np

# 1. Load the hand STL
# Ensure the path points to your hand mesh file
mesh = trimesh.load('/media/yiting/NewVolume/Data/Hand_anatomy/MRI/Neo Hand Segmentation v1 repaired.stl')

# 2. Load JSON coordinates of keypoints
# Expected format: {"Index_Tip": [x, y, z], ...}
with open('/home/yiting/Documents/GitHub/hand_tracking/configs/hand_keypoint_map.json', 'r') as f:
    keypoints_data = json.load(f)

# 3. Create a scene to hold the mesh and markers
scene = trimesh.Scene()
scene.add_geometry(mesh)

# 4. Create markers (spheres) for each keypoint
marker_radius = 2.0  # mm (adjust size for visibility)
marker_color = [255, 0, 0, 255]  # Red

for name, pos in keypoints_data.items():
    # Create a small sphere at the coordinate
    sphere = trimesh.creation.uv_sphere(radius=marker_radius)
    sphere.visual.face_colors = marker_color
    
    # Translate the sphere to the keypoint position
    translation = np.eye(4)
    translation[:3, 3] = pos
    sphere.apply_transform(translation)
    
    # Add sphere and its name to the scene
    scene.add_geometry(sphere, node_name=f"marker_{name}")

# 5. Launch interactive visualization
# This will open a window showing the mesh with red dots
print("Opening interactive viewer...")
scene.show()