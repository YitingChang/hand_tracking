import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer

# --- 1. CONFIGURATION ---
MESH_DIR = "/home/yiting/Documents/GitHub/myo_sim/meshes"
TEXTURE_DIR = "/home/yiting/Documents/GitHub/myo_sim/texture"
FRAME_IDX = 300

def sanitize_assets_file(assets_path):
    """Removes compiler tags and strips redundant path prefixes from meshes."""
    tree = ET.parse(assets_path)
    root = tree.getroot()
    for compiler in root.findall('compiler'):
        root.remove(compiler)
    for mesh in root.findall('.//mesh'):
        file_path = mesh.get('file')
        if file_path and '/' in file_path:
            mesh.set('file', file_path.split('/')[-1])
    tree.write(assets_path)

def calibrate_myohand(csv_path, xml_input_path, xml_output_path, frame_idx=300):
    # 1. Load Tracking Data
    df = pd.read_csv(csv_path)
    f = df.iloc[frame_idx]
    def get_xyz(prefix):
        return np.array([f[f'{prefix}_x'], f[f'{prefix}_y'], f[f'{prefix}_z']]) / 1000.0

    wrist = (get_xyz('Wrist_U') + get_xyz('Wrist_R')) / 2.0
    finger_map = {
        'index':  {'pts': ['Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip'], 'id': '2', 'tip': 'IFtip'},
        'middle': {'pts': ['Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip'], 'id': '3', 'tip': 'MFtip'},
        'ring':   {'pts': ['Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip'], 'id': '4', 'tip': 'RFtip'},
        'pinky':  {'pts': ['Small_MCP', 'Small_PIP', 'Small_DIP', 'Small_Tip'], 'id': '5', 'tip': 'LFtip'}
    }
    thumb_pts = ['Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip']

    # 2. Calculation logic
    def dist(p1, p2): return np.linalg.norm(p1 - p2)
    def get_angle(v1, v2):
        u1, u2 = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))

    bone_calib, joint_angles = {}, {}
    for name, data in finger_map.items():
        idx, p = data['id'], [get_xyz(pt) for pt in data['pts']]
        bone_calib[f'midph{idx}'], bone_calib[f'distph{idx}'], bone_calib[data['tip']] = dist(p[0],p[1]), dist(p[1],p[2]), dist(p[2],p[3])
        v_met, v_prox, v_mid, v_dist = p[0]-wrist, p[1]-p[0], p[2]-p[1], p[3]-p[2]
        joint_angles[f'mcp{idx}_flexion'], joint_angles[f'pm{idx}_flexion'], joint_angles[f'md{idx}_flexion'] = get_angle(v_met,v_prox), get_angle(v_prox,v_mid), get_angle(v_mid,v_dist)

    t = [get_xyz(pt) for pt in thumb_pts]
    bone_calib['proximal_thumb'], bone_calib['distal_thumb'], bone_calib['THtip'] = dist(t[0],t[1]), dist(t[1],t[2]), dist(t[2],t[3])
    v_th_cmc, v_th_prox, v_th_mid, v_th_dist = t[0]-wrist, t[1]-t[0], t[2]-t[1], t[3]-t[2]
    joint_angles['cmc_flexion'], joint_angles['mp_flexion'], joint_angles['ip_flexion'] = get_angle(v_th_cmc,v_th_prox), get_angle(v_th_prox,v_th_mid), get_angle(v_th_mid,v_th_dist)

    # 3. Fix XML Structure and Geometry Types
    tree = ET.parse(xml_input_path)
    root = tree.getroot()
    new_root = ET.Element('mujoco', model="monkey_hand")
    ET.SubElement(new_root, 'compiler', meshdir=MESH_DIR, texturedir=TEXTURE_DIR, balanceinertia="true")
    ET.SubElement(new_root, 'include', file="myohand_assets.xml")
    worldbody = ET.SubElement(new_root, 'worldbody')

    # Important: Flattening classes while preserving hierarchy
    for child in list(root):
        if 'childclass' in child.attrib: del child.attrib['childclass']
        for sub in child.iter():
            # Hide the "balls" (wrap geoms and markers)
            if sub.tag == 'geom':
                # Check if it's a wrap geom (often spheres in MyoHand)
                if 'wrap' in sub.get('name', '').lower():
                    sub.set('rgba', '0 0 0 0') # Make invisible
                
                # Fix fromto geoms lost by stripping classes
                if 'fromto' in sub.attrib and 'type' not in sub.attrib:
                    sub.set('type', 'capsule')
            
            # Move sites (markers) to group 4 so they can be toggled off
            if sub.tag == 'site':
                sub.set('group', '4')

            if 'class' in sub.attrib: del sub.attrib['class']
        worldbody.append(child)

    # 4. Apply Calibration with Axis Preservation
    for name, length in bone_calib.items():
        tag = 'site' if 'tip' in name else 'body'
        # Search globally in the new_root for the element
        elem = new_root.find(f".//{tag}[@name='{name}']")
        
        if elem is not None:
            old_pos_str = elem.get('pos')
            if old_pos_str:
                old_pos = np.array([float(x) for x in old_pos_str.split()])
                
                # Identify the dominant axis (usually bones extend along one primary axis)
                if np.linalg.norm(old_pos) > 1e-6:
                    # Find which axis (0, 1, or 2) has the largest value
                    main_axis = np.argmax(np.abs(old_pos))
                    
                    # Create new position: preserve original tilt but match monkey length
                    new_pos = (old_pos / np.linalg.norm(old_pos)) * length
                    elem.set('pos', f"{new_pos[0]:.6f} {new_pos[1]:.6f} {new_pos[2]:.6f}")
                else:
                    # If original pos was 0 (like a joint at origin), 
                    # we must use a default bone direction (usually negative Y in MyoHand)
                    elem.set('pos', f"0 {-length:.6f} 0")

    ET.ElementTree(new_root).write(xml_output_path)
    return joint_angles

def main():
    session_dir = Path('/media/yiting/NewVolume/Analysis/2025-12-09')
    trial_name = '2025-12-09_09-02-01'
    csv_path = session_dir / 'anipose' / 'pose_3d_filter' / f'{trial_name}_f3d.csv'
    myo_sim_dir = Path('/home/yiting/Documents/GitHub/myo_sim')
    body_xml_path = myo_sim_dir / 'hand' / 'assets' / 'myohand_body.xml'
    original_assets_path = myo_sim_dir / 'hand' / 'assets' / 'myohand_assets.xml'
    output_dir = session_dir / 'mujoco' / 'calibrated_xml' / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_xml_path = output_dir / 'myohand_monkey_calibrated.xml'

    local_assets_path = output_dir / 'myohand_assets.xml'
    shutil.copy(original_assets_path, local_assets_path)
    sanitize_assets_file(local_assets_path)

    joint_angles = calibrate_myohand(csv_path, body_xml_path, calibrated_xml_path, FRAME_IDX)
    model = mujoco.MjModel.from_xml_path(str(calibrated_xml_path))
    data = mujoco.MjData(model)

    for name, angle in joint_angles.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1: data.qpos[model.jnt_qposadr[jid]] = angle
    
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running(): viewer.sync()

if __name__ == "__main__": main()