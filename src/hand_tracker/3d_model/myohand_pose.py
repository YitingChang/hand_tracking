import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer

# --- CONFIGURATION ---
MESH_DIR = "/home/yiting/Documents/GitHub/myo_sim/meshes"
TEXTURE_DIR = "/home/yiting/Documents/GitHub/myo_sim/texture"
FRAME_IDX = 300

def sanitize_assets_file(assets_path):
    tree = ET.parse(assets_path)
    root = tree.getroot()
    for compiler in root.findall('compiler'):
        root.remove(compiler)
    for mesh in root.findall('.//mesh'):
        file_path = mesh.get('file')
        if file_path and '/' in file_path:
            mesh.set('file', file_path.split('/')[-1])
    tree.write(assets_path)

def debug_calibrate_angles(csv_path, xml_input_path, xml_output_path, frame_idx=300):
    # 1. Load Tracking Data
    df = pd.read_csv(csv_path)
    f = df.iloc[frame_idx]
    def get_xyz(prefix):
        return np.array([f[f'{prefix}_x'], f[f'{prefix}_y'], f[f'{prefix}_z']]) / 1000.0

    wrist = (get_xyz('Wrist_U') + get_xyz('Wrist_R')) / 2.0
    
    finger_map = {
        'index':  {'pts': ['Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip'], 'id': '2'},
        'middle': {'pts': ['Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip'], 'id': '3'},
        'ring':   {'pts': ['Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip'], 'id': '4'},
        'pinky':  {'pts': ['Small_MCP', 'Small_PIP', 'Small_DIP', 'Small_Tip'], 'id': '5'}
    }
    thumb_pts = ['Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip']

    def get_angle(v1, v2):
        u1, u2 = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))

    joint_angles = {}
    for name, data in finger_map.items():
        idx, p = data['id'], [get_xyz(pt) for pt in data['pts']]
        v_met, v_prox, v_mid, v_dist = p[0]-wrist, p[1]-p[0], p[2]-p[1], p[3]-p[2]
        # MCP, PIP, DIP mapping
        joint_angles[f'mcp{idx}_flexion'] = get_angle(v_met, v_prox)
        joint_angles[f'pm{idx}_flexion']  = get_angle(v_prox, v_mid)
        joint_angles[f'md{idx}_flexion']  = get_angle(v_mid, v_dist)

    t = [get_xyz(pt) for pt in thumb_pts]
    v_th_cmc, v_th_prox, v_th_mid, v_th_dist = t[0]-wrist, t[1]-t[0], t[2]-t[1], t[3]-t[2]
    joint_angles['cmc_flexion'] = get_angle(v_th_cmc, v_th_prox)
    joint_angles['mp_flexion']  = get_angle(v_th_prox, v_th_mid)
    joint_angles['ip_flexion']  = get_angle(v_th_mid, v_th_dist)

    # 2. Fix Standalone XML Structure (NO POS UPDATES)
    tree = ET.parse(xml_input_path)
    root = tree.getroot()
    new_root = ET.Element('mujoco', model="monkey_hand_debug")
    ET.SubElement(new_root, 'compiler', meshdir=MESH_DIR, texturedir=TEXTURE_DIR, balanceinertia="true")
    ET.SubElement(new_root, 'include', file="myohand_assets.xml")
    worldbody = ET.SubElement(new_root, 'worldbody')

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

    ET.ElementTree(new_root).write(xml_output_path)
    return joint_angles

def main():
    session_dir = Path('/media/yiting/NewVolume/Analysis/2025-12-09')
    trial_name = '2025-12-09_09-02-01'
    csv_path = session_dir / 'anipose' / 'pose_3d_filter' / f'{trial_name}_f3d.csv'
    myo_sim_dir = Path('/home/yiting/Documents/GitHub/myo_sim')
    body_xml_path = myo_sim_dir / 'hand' / 'assets' / 'myohand_body.xml'
    original_assets_path = myo_sim_dir / 'hand' / 'assets' / 'myohand_assets.xml'
    output_dir = session_dir / 'mujoco' / 'debug_xml'
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_xml_path = output_dir / 'myohand_debug_angles.xml'

    shutil.copy(original_assets_path, output_dir / 'myohand_assets.xml')
    sanitize_assets_file(output_dir / 'myohand_assets.xml')

    # Run only Angle Calculation
    joint_angles = debug_calibrate_angles(csv_path, body_xml_path, debug_xml_path, FRAME_IDX)
    model = mujoco.MjModel.from_xml_path(str(debug_xml_path))
    data = mujoco.MjData(model)

    for name, angle in joint_angles.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1: 
            data.qpos[model.jnt_qposadr[jid]] = angle
    
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running(): viewer.sync()

if __name__ == "__main__":
    main()