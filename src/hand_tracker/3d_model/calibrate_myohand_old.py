from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import shutil

# Set paths
MESH_DIR = "/home/yiting/Documents/GitHub/myo_sim/meshes"
TEXTURE_DIR = "/home/yiting/Documents/GitHub/myo_sim/texture"
# ASSETS_DIR = "/home/yiting/Documents/GitHub/myo_sim/hand/assets" # Folder containing myohand_assets.xml

FRAME_IDX = 300

def sanitize_assets_file(assets_path):
    """Removes compiler tags and cleans relative file paths in assets."""
    tree = ET.parse(assets_path)
    root = tree.getroot()
    
    # 1. Remove compiler tags
    for compiler in root.findall('compiler'):
        root.remove(compiler)
        
    # 2. Strip redundant folder prefixes from mesh file paths
    # This changes '../myo_sim/meshes/hand.stl' to 'hand.stl'
    for mesh in root.findall('.//mesh'):
        file_path = mesh.get('file')
        if file_path and '/' in file_path:
            filename = file_path.split('/')[-1]
            mesh.set('file', filename)
            
    tree.write(assets_path)

def calibrate_myohand(csv_path, xml_input_path, xml_output_path, frame_idx=300):
    # 1. Load Tracking Data
    df = pd.read_csv(csv_path)
    f = df.iloc[frame_idx]

    def get_xyz(prefix):
        return np.array([f[f'{prefix}_x'], f[f'{prefix}_y'], f[f'{prefix}_z']]) / 1000.0

    # 2. Extract Keypoints
    wrist = (get_xyz('Wrist_U') + get_xyz('Wrist_R')) / 2.0
    
    finger_map = {
        'index':  {'pts': ['Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip'], 'id': '2', 'tip': 'IFtip'},
        'middle': {'pts': ['Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip'], 'id': '3', 'tip': 'MFtip'},
        'ring':   {'pts': ['Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip'], 'id': '4', 'tip': 'RFtip'},
        'pinky':  {'pts': ['Small_MCP', 'Small_PIP', 'Small_DIP', 'Small_Tip'], 'id': '5', 'tip': 'LFtip'}
    }
    thumb_pts = ['Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip']

    def dist(p1, p2): return np.linalg.norm(p1 - p2)
    def get_angle(v1, v2):
        u1, u2 = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))

    # 3. Calculate Calibrations
    bone_calib = {}
    joint_angles = {}

    for name, data in finger_map.items():
        idx = data['id']
        p = [get_xyz(pt) for pt in data['pts']]
        
        # Lengths
        bone_calib[f'midph{idx}']  = dist(p[0], p[1]) 
        bone_calib[f'distph{idx}'] = dist(p[1], p[2]) 
        bone_calib[data['tip']]    = dist(p[2], p[3]) 
        
        # Angles - Mapping keys EXACTLY to MuJoCo Joint Names
        v_met, v_prox, v_mid, v_dist = p[0]-wrist, p[1]-p[0], p[2]-p[1], p[3]-p[2]
        joint_angles[f'mcp{idx}_flexion'] = get_angle(v_met, v_prox)
        joint_angles[f'pm{idx}_flexion']  = get_angle(v_prox, v_mid)
        joint_angles[f'md{idx}_flexion']  = get_angle(v_mid, v_dist)

    t = [get_xyz(pt) for pt in thumb_pts]
    bone_calib['proximal_thumb'] = dist(t[0], t[1])
    bone_calib['distal_thumb']   = dist(t[1], t[2])
    bone_calib['THtip']          = dist(t[2], t[3])
    
    v_th_cmc, v_th_prox, v_th_mid, v_th_dist = t[0]-wrist, t[1]-t[0], t[2]-t[1], t[3]-t[2]
    joint_angles['cmc_flexion'] = get_angle(v_th_cmc, v_th_prox)
    joint_angles['mp_flexion']  = get_angle(v_th_prox, v_th_mid)
    joint_angles['ip_flexion']  = get_angle(v_th_mid, v_th_dist)

    # 4. Build Standalone XML
    tree = ET.parse(xml_input_path)
    root = tree.getroot()
    new_root = ET.Element('mujoco', model="monkey_hand")

    # Set paths to actual local folders
    ET.SubElement(new_root, 'compiler', meshdir=MESH_DIR, texturedir=TEXTURE_DIR)
    
    # Include the asset definitions (ensure this file is in your output directory)
    ET.SubElement(new_root, 'include', file="myohand_assets.xml")
    
    worldbody = ET.SubElement(new_root, 'worldbody')

    # Clean up and append the body segments
    for child in list(root):
        if 'childclass' in child.attrib:
            del child.attrib['childclass']
        for sub in child.iter():
            if 'class' in sub.attrib:
                del sub.attrib['class']
        worldbody.append(child)

    def update_xml_pos(target_name, new_length, tag='body'):
        elem = root.find(f".//{tag}[@name='{target_name}']")
        if elem is not None:
            old_pos = np.array([float(x) for x in elem.get('pos').split()])
            if np.linalg.norm(old_pos) > 0:
                new_pos = (old_pos / np.linalg.norm(old_pos)) * new_length
                elem.set('pos', f"{new_pos[0]:.6f} {new_pos[1]:.6f} {new_pos[2]:.6f}")

    for name, length in bone_calib.items():
        update_xml_pos(name, length, tag='site' if 'tip' in name.lower() else 'body')

    tree = ET.ElementTree(new_root)
    tree.write(xml_output_path)

    return joint_angles

def set_myohand_qpos(model_path, joint_angles):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # All joint_angles keys now match the XML names exactly
    for joint_name, angle in joint_angles.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            data.qpos[model.jnt_qposadr[joint_id]] = angle
        else:
            print(f"Warning: Joint '{joint_name}' not found.")

    mujoco.mj_forward(model, data)
    return model, data

def main():
    # 0. File Paths
    session_dir = Path('/media/yiting/NewVolume/Analysis/2025-12-09')
    pose_3d_dir = session_dir / 'anipose' / 'pose_3d_filter'
    trial_name = '2025-12-09_09-02-01'
    csv_path = pose_3d_dir / f'{trial_name}_f3d.csv'

    # Paths to ORIGINAL MyoSim files
    myo_sim_dir = Path('/home/yiting/Documents/GitHub/myo_sim')
    body_xml_path = myo_sim_dir / 'hand' / 'assets' / 'myohand_body.xml'
    original_assets_path = myo_sim_dir / 'hand' / 'assets' / 'myohand_assets.xml'

    # Output directory for THIS frame
    output_dir = session_dir / 'mujoco' / 'calibrated_xml' / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_xml_path = output_dir / 'myohand_monkey_calibrated.xml'
    
    # 1. Copy and Sanitize the assets file FIRST
    local_assets_path = output_dir / 'myohand_assets.xml'
    shutil.copy(original_assets_path, local_assets_path)
    sanitize_assets_file(local_assets_path)

    # 2. Calibrate and generate the XML
    joint_angles = calibrate_myohand(csv_path, body_xml_path, calibrated_xml_path, frame_idx=FRAME_IDX)

    # 3. Load and Visualize
    model = mujoco.MjModel.from_xml_path(str(calibrated_xml_path))
    data = mujoco.MjData(model)

    # Set initial posture
    for joint_name, angle in joint_angles.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            data.qpos[model.jnt_qposadr[joint_id]] = angle
    
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()