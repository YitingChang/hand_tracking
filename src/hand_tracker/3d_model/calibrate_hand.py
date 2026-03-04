import os
from pathlib import Path
import json
import numpy as np

# Load Frame 300 JSON
data_path = Path('/media/yiting/NewVolume/Analysis/2025-12-09/mujoco/data/frame_300_keypoints.json')

with open(data_path, 'r') as f:
    data = json.load(f)
hand = data['hand']


# Functions
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_angle(v1, v2):
    """Calculates the 1-DoF flexion/extension angle between segments."""
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))


# ------ Full Bone Length Calibration (pos) ------
# Bone Lengths for XML 'pos' (Offsets relative to parent body)
# MyoHand fingers: Index(pro, mid, dist), Middle(pro, mid, dist), etc.

calib_bones = {}

for f_name in ['index', 'middle', 'ring', 'pinky']:
    calib_bones[f'{f_name}_metacarpal'] = dist(hand['wrist'], hand[f_name]['mcp'])
    calib_bones[f'{f_name}_proximal']  = dist(hand[f_name]['mcp'], hand[f_name]['pip'])
    calib_bones[f'{f_name}_middle']    = dist(hand[f_name]['pip'], hand[f_name]['dip'])
    calib_bones[f'{f_name}_distal']    = dist(hand[f_name]['dip'], hand[f_name]['tip'])

# Thumb: CMC -> MCP -> IP -> Tip
calib_bones['thumb_cmc_to_mcp'] = dist(hand['thumb']['cmc'], hand['thumb']['mcp'])
calib_bones['thumb_mcp_to_ip']  = dist(hand['thumb']['mcp'], hand['thumb']['ip'])
calib_bones['thumb_ip_to_tip']  = dist(hand['thumb']['ip'],  hand['thumb']['tip'])

print("--- MyoHand XML Calibrations (Meters) ---")
for k, v in calib_bones.items():
    print(f"{k:25}: {v:.4f}")


# ------ Full Joint Angle Calibration (qpos) ------

qpos_results = {}

for f_name in ['index', 'middle', 'ring', 'pinky']:
    # Vectors
    v_met = np.array(hand[f_name]['mcp']) - np.array(hand['wrist'])
    v_pro = np.array(hand[f_name]['pip']) - np.array(hand[f_name]['mcp'])
    v_mid = np.array(hand[f_name]['dip']) - np.array(hand[f_name]['pip'])
    v_dis = np.array(hand[f_name]['tip']) - np.array(hand[f_name]['dip'])
    
    # Angles
    qpos_results[f'{f_name}_mcp_flex'] = get_angle(v_met, v_pro)
    qpos_results[f'{f_name}_pip_flex'] = get_angle(v_pro, v_mid)
    qpos_results[f'{f_name}_dip_flex'] = get_angle(v_mid, v_dis)

# Thumb IP and MCP flexion
v_th_cmc = np.array(hand['thumb']['mcp']) - np.array(hand['thumb']['cmc'])
v_th_mcp = np.array(hand['thumb']['ip'])  - np.array(hand['thumb']['mcp'])
v_th_ip  = np.array(hand['thumb']['tip']) - np.array(hand['thumb']['ip'])

qpos_results['thumb_mcp_flex'] = get_angle(v_th_cmc, v_th_mcp)
qpos_results['thumb_ip_flex']  = get_angle(v_th_mcp, v_th_ip)

print("\n--- Joint Posture (Radians) ---")
for k, v in qpos_results.items():
    print(f"{k:20}: {v:.4f}")

# ------ Anatomical Thinning (Monkey Hand) ------
# The MyoHand uses detailed bone meshes (.stl). To make it look thinner:

# Global Scale: In your XML's <worldbody>, find the hand body and set a scale (e.g., 0.7 0.7 0.7) to shrink it to monkey size.

# Visual Thinning: Change the <mesh> asset scale for the finger bones.

# Example: <mesh file="proximal_index.stl" scale="0.6 0.6 1.0"/>. This keeps the bone length (Z) but thins the width (X, Y).