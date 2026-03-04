import json
import numpy as np
from pathlib import Path

# Load Frame 300 JSON
data_path = Path('/media/yiting/NewVolume/Analysis/2025-12-09/mujoco/data/frame_300_keypoints.json')
with open(data_path, 'r') as f:
    data = json.load(f)

hand = data['hand']

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 1. Calculate Bone Lengths for XML 'pos' attributes
bone_lengths = {}
for finger in ['index', 'middle', 'ring', 'pinky']:
    # Distance from MCP to PIP (Proximal)
    bone_lengths[f'{finger}_prox'] = dist(hand[finger]['mcp'], hand[finger]['pip'])
    # Distance from PIP to DIP (Middle)
    bone_lengths[f'{finger}_mid']  = dist(hand[finger]['pip'], hand[finger]['dip'])
    # Distance from DIP to Tip (Distal)
    bone_lengths[f'{finger}_dist'] = dist(hand[finger]['dip'], hand[finger]['tip'])

# Thumb specific (CMC -> MCP -> IP -> Tip)
bone_lengths['thumb_prox'] = dist(hand['thumb']['cmc'], hand['thumb']['mcp'])
bone_lengths['thumb_mid']  = dist(hand['thumb']['mcp'], hand['thumb']['ip'])
bone_lengths['thumb_dist'] = dist(hand['thumb']['ip'],  hand['thumb']['tip'])

print("--- XML Calibration (Meters) ---")
for k, v in bone_lengths.items():
    print(f"{k:15}: {v:.4f}")


# 2. Calculate Angles for XML 'qpos' attributes

def get_angle(v1, v2):
    """Calculates flexion angle between two bone vectors."""
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))

qpos_angles = {}

for finger in ['index', 'middle', 'ring', 'pinky']:
    v_p = np.array(hand[finger]['pip']) - np.array(hand[finger]['mcp'])
    v_m = np.array(hand[finger]['dip']) - np.array(hand[finger]['pip'])
    v_d = np.array(hand[finger]['tip']) - np.array(hand[finger]['dip'])
    
    # J2: PIP Flexion (Angle between Proximal and Middle)
    qpos_angles[f'{finger}_J2'] = get_angle(v_p, v_m)
    # J1: DIP Flexion (Angle between Middle and Distal)
    qpos_angles[f'{finger}_J1'] = get_angle(v_m, v_d)

# Thumb Flexion
v_th_p = np.array(hand['thumb']['mcp']) - np.array(hand['thumb']['cmc'])
v_th_m = np.array(hand['thumb']['ip'])  - np.array(hand['thumb']['mcp'])
v_th_d = np.array(hand['thumb']['tip']) - np.array(hand['thumb']['ip'])

qpos_angles['thumb_J3'] = get_angle(v_th_p, v_th_m) # MCP flexion
qpos_angles['thumb_J2'] = get_angle(v_th_m, v_th_d) # IP flexion

print("\n--- Joint Angles (Radians for qpos) ---")
for k, v in qpos_angles.items():
    print(f"{k:15}: {v:.4f}")