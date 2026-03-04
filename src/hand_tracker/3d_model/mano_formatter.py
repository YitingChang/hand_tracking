
from pathlib import Path
import os
import pandas as pd
import numpy as np

FRAME_NUMBER = 100

def extract_mano_frame(csv_path, frame_idx=FRAME_NUMBER):
    df = pd.read_csv(csv_path)
    f = df.iloc[frame_idx]

    def get_xyz(prefix):
        return [f[f'{prefix}_x'], f[f'{prefix}_y'], f[f'{prefix}_z']]

    # Calculate Wrist (Root) as midpoint of the two wrist markers
    wrist = (np.array(get_xyz('Wrist_U')) + np.array(get_xyz('Wrist_R'))) / 2.0

    # Map our custom labels to MANO 21-joint order
    # (Wrist, Index 1-3, Middle 1-3, Pinky 1-3, Ring 1-3, Thumb 1-3, Tips 16-20)
    mano_joints = np.array([
        wrist,                          # 0: Wrist
        get_xyz('Index_MCP'),           # 1
        get_xyz('Index_PIP'),           # 2
        get_xyz('Index_DIP'),           # 3
        get_xyz('Middle_MCP'),          # 4
        get_xyz('Middle_PIP'),          # 5
        get_xyz('Middle_DIP'),          # 6
        get_xyz('Small_MCP'),           # 7
        get_xyz('Small_PIP'),           # 8
        get_xyz('Small_DIP'),           # 9
        get_xyz('Ring_MCP'),            # 10
        get_xyz('Ring_PIP'),            # 11
        get_xyz('Ring_DIP'),            # 12
        get_xyz('Thumb_CMC'),           # 13
        get_xyz('Thumb_MCP'),           # 14
        get_xyz('Thumb_IP'),            # 15
        get_xyz('Index_Tip'),           # 16
        get_xyz('Middle_Tip'),          # 17
        get_xyz('Small_Tip'),           # 18
        get_xyz('Ring_Tip'),            # 19
        get_xyz('Thumb_Tip'),           # 20
    ])
    
    return mano_joints

# Usage
data_dir = Path('/media/yiting/NewVolume/Analysis/2025-08-19/anipose/pose_3d_filter')
trial_name = '2025-08-19_08-48-24'
csv_path = data_dir / f'{trial_name}_f3d.csv'
keypoints_mm = extract_mano_frame(csv_path, FRAME_NUMBER)

# IMPORTANT: Convert to meters if your MANO model expects meters
keypoints_m = keypoints_mm / 1000.0

mano_dir = Path('/media/yiting/NewVolume/Analysis/2025-08-19/mano')
np.save(mano_dir / f'mano_{trial_name}_frame{FRAME_NUMBER}.npy', keypoints_m)
