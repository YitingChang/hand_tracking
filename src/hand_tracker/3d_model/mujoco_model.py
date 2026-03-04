from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

# --- 1. SETUP PATHS & DATA ---
frame_idx = 300
# Update this to your actual directory structure
session_dir = Path('/media/yiting/NewVolume/Analysis/2025-12-09')
data_path = session_dir / 'mujoco' / 'data' / f'frame_{frame_idx}_keypoints.json'

with open(data_path, 'r') as f:
    keypoints_data = json.load(f)

# Extract nested JSON data
hand_dict = keypoints_data['hand']
wrist_pos = np.array(hand_dict['wrist'])

# Helper to flatten the nested finger joints into a list of 21 points
def flatten_hand(d):
    pts = [d['wrist']] # Start with wrist (Index 0)
    # Order matches the JSON: index, middle, pinky, ring, thumb
    for finger in ['index', 'middle', 'pinky', 'ring', 'thumb']:
        for joint in d[finger].values():
            pts.append(joint)
    return np.array(pts)

# hand_pts_m is now a (21, 3) array in meters
hand_pts_m = flatten_hand(hand_dict)

# --- 2. OBJECT ALIGNMENT (Commented out for later) ---
"""
# Local Coordinates of Dots on the Physical Object (relative to STL origin)
dots_local = np.array([[...]]) 
dots_tracked = np.array([v for v in keypoints_data['object_markers'].values()])

def find_rigid_transform(A, B):
    centroid_A, centroid_B = np.mean(A, axis=0), np.mean(B, axis=0)
    H = (A - centroid_A).T @ (B - centroid_B)
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[2,:] *= -1
        R_mat = Vt.T @ U.T
    return R_mat, centroid_B - R_mat @ centroid_A

obj_rot, obj_pos = find_rigid_transform(dots_local, dots_tracked)
obj_quat = R.from_matrix(obj_rot).as_quat() # [x y z w]
"""

# --- 3. GENERATE MUJOCO XML (MJCF) ---
# Removed references to obj_pos and obj_quat since they are currently disabled
mjcf_template = f"""
<mujoco model="monkey_grasp">
    <asset>
        <mesh name="printed_object" file="printed_object.stl" scale="0.001 0.001 0.001"/>
    </asset>
    
    <worldbody>
        <light directional="true" pos="-0.5 0.5 3" dir="0 0 -1"/>

        <body name="hand_target" mocap="true" pos="{wrist_pos[0]} {wrist_pos[1]} {wrist_pos[2]}">
            <geom type="sphere" size="0.01" rgba="0 1 0 0.5"/>
        </body>
        
        <body name="index_tip_target" mocap="true" pos="{hand_dict['index']['tip'][0]} {hand_dict['index']['tip'][1]} {hand_dict['index']['tip'][2]}">
            <geom type="sphere" size="0.005" rgba="1 0 0 0.5"/>
        </body>

        <include file="adroit_hand_scaled.xml"/>
    </worldbody>
</mujoco>
"""

output_path = session_dir / 'mujoco' / 'monkey_scene.xml'
with open(output_path, 'w') as f:
    f.write(mjcf_template)

print(f"MuJoCo XML successfully generated at {output_path}")
