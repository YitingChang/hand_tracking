from pathlib import Path
import pandas as pd
import numpy as np
import json

# Paths and filenames
session_dir = Path('/media/yiting/NewVolume/Analysis/2025-12-09')
pose_3d_dir = session_dir / 'anipose' / 'pose_3d_filter'
mujoco_data_dir = session_dir / 'mujoco' / 'data'
mujoco_data_dir.mkdir(parents=True, exist_ok=True)

# Load your tracking data
csv_file = '2025-12-09_09-02-01_f3d.csv'
csv_path = pose_3d_dir / csv_file
df = pd.read_csv(csv_path)

# Extract Frame 300
frame_idx = 300
f300 = df.iloc[frame_idx]

def get_xyz_meters(prefix):
    """Returns XYZ as a numpy array in meters (MuJoCo default)."""
    return np.array([f300[f'{prefix}_x'], f300[f'{prefix}_y'], f300[f'{prefix}_z']]) / 1000.0

# 1. Organize Hand Keypoints (21 joints) mapped to Adroit/Human convention
hand_data = {
    "wrist": (get_xyz_meters('Wrist_U') + get_xyz_meters('Wrist_R')) / 2.0,
    "index": {
        "mcp": get_xyz_meters('Index_MCP'), # Maps to Adroit FFJ3
        "pip": get_xyz_meters('Index_PIP'), # Maps to Adroit FFJ2
        "dip": get_xyz_meters('Index_DIP'), # Maps to Adroit FFJ1
        "tip": get_xyz_meters('Index_Tip')
    },
    "middle": {
        "mcp": get_xyz_meters('Middle_MCP'), # Maps to Adroit MFJ3
        "pip": get_xyz_meters('Middle_PIP'), # Maps to Adroit MFJ2
        "dip": get_xyz_meters('Middle_DIP'), # Maps to Adroit MFJ1
        "tip": get_xyz_meters('Middle_Tip')
    },
    "ring": {
        "mcp": get_xyz_meters('Ring_MCP'),   # Maps to Adroit RFJ3
        "pip": get_xyz_meters('Ring_PIP'),   # Maps to Adroit RFJ2
        "dip": get_xyz_meters('Ring_DIP'),   # Maps to Adroit RFJ1
        "tip": get_xyz_meters('Ring_Tip')
    },
    "pinky": {
        "mcp": get_xyz_meters('Small_MCP'),  # Maps to Adroit LFJ3
        "pip": get_xyz_meters('Small_PIP'),  # Maps to Adroit LFJ2
        "dip": get_xyz_meters('Small_DIP'),  # Maps to Adroit LFJ1
        "tip": get_xyz_meters('Small_Tip')
    },
    "thumb": {
        "cmc": get_xyz_meters('Thumb_CMC'),  # Maps to Adroit THJ4
        "mcp": get_xyz_meters('Thumb_MCP'),  # Maps to Adroit THJ3
        "ip":  get_xyz_meters('Thumb_IP'),   # Maps to Adroit THJ2
        "tip": get_xyz_meters('Thumb_Tip')   # Maps to Adroit THJ1
    }
}

# 2. Organize Object Markers (12 dots)
dot_prefixes = ['Dot_t1', 'Dot_t2', 'Dot_t3', 'Dot_b1', 'Dot_b2', 'Dot_b3', 
                'Dot_l1', 'Dot_l2', 'Dot_l3', 'Dot_r1', 'Dot_r2', 'Dot_r3']
object_markers = {dot: get_xyz_meters(dot).tolist() for dot in dot_prefixes}

# 3. Save as JSON for your MuJoCo script
def numpy_to_list(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: numpy_to_list(v) for k, v in obj.items()}
    return obj

final_data = {
    "frame": frame_idx,
    "hand": numpy_to_list(hand_data),
    "object_markers": object_markers
}

output_path = mujoco_data_dir / f'frame_{frame_idx}_keypoints.json'
with open(output_path, 'w') as f:
    json.dump(final_data, f, indent=4)

print(f"Extraction complete for Frame {frame_idx}. Data saved to {output_path}")