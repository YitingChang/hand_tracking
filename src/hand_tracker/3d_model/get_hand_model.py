from pathlib import Path
import json
import numpy as np
import mujoco

# Assuming 'qpos_results' is the dictionary we calculated in the previous turn
# and 'model_path' points to your scaled MyoHand XML.

model_path = Path('/home/yiting/Documents/GitHub/myo_sim/hand/myohand.xml')

def apply_frame_300_to_myo(model_path, qpos_results):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 1. Reset all joints to zero
    data.qpos[:] = 0.0
    
    # 2. Map calculated angles to specific joint names in MyoHand
    # We use joint names to find indices to avoid hardcoding numbers
    joint_mapping = {
        'index_mcp':  qpos_results['index_mcp_flex'],
        'index_pip':  qpos_results['index_pip_flex'],
        'index_dip':  qpos_results['index_dip_flex'],
        'middle_mcp': qpos_results['middle_mcp_flex'],
        'middle_pip': qpos_results['middle_pip_flex'],
        'middle_dip': qpos_results['middle_dip_flex'],
        'ring_mcp':   qpos_results['ring_mcp_flex'],
        'ring_pip':   qpos_results['ring_pip_flex'],
        'ring_dip':   qpos_results['ring_dip_flex'],
        'pinky_mcp':  qpos_results['pinky_mcp_flex'],
        'pinky_pip':  qpos_results['pinky_pip_flex'],
        'pinky_dip':  qpos_results['pinky_dip_flex'],
        'thumb_mcp':  qpos_results['thumb_mcp_flex'],
        'thumb_ip':   qpos_results['thumb_ip_flex']
    }

    for j_name, angle in joint_mapping.items():
        try:
            # MyoHand joints often have suffixes like '_joint' or '_flexion'
            # Adjust the search string based on your specific XML names
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = angle
        except:
            print(f"Warning: Joint {j_name} not found in model.")

    # 3. Propagate kinematics
    mujoco.mj_forward(model, data)
    
    return model, data

# --- Execution ---
# model, data = apply_frame_300_to_myo('myo_hand_scaled.xml', qpos_results)