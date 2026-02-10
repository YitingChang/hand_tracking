import trimesh
import pyrender
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from glob import glob

# --- CONFIGURATION ---
STL_INPUT_DIR = "/media/yiting/NewVolume/Data/Shapes/shapes_stl"
RENDER_OUTPUT_DIR = "/media/yiting/NewVolume/Data/Shapes/shapes_6views_ori2"
Z_CUTOFF = -30.0  # Adjust this to the height where the stem meets the base
ORIENTATION = 90  # Degrees to rotate around Z-axis (stem) before slicing, e.g., 0, 90, 180, etc.

def get_6view_poses(distance):
    """Generates 4x4 transformation matrices for 6 cardinal directions."""
    poses = {}
    # Forward (Front)
    poses['Front'] = trimesh.transformations.translation_matrix([0, -distance, 0]) @ \
                     trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0])
    # Backward (Back)
    poses['Back'] = trimesh.transformations.translation_matrix([0, distance, 0]) @ \
                    trimesh.transformations.rotation_matrix(np.deg2rad(-90), [1, 0, 0]) @ \
                    trimesh.transformations.rotation_matrix(np.deg2rad(180), [0, 0, 1])
    # Left
    poses['Left'] = trimesh.transformations.translation_matrix([-distance, 0, 0]) @ \
                    trimesh.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0]) @ \
                    trimesh.transformations.rotation_matrix(np.deg2rad(-90), [0, 0, 1])
    # Right
    poses['Right'] = trimesh.transformations.translation_matrix([distance, 0, 0]) @ \
                     trimesh.transformations.rotation_matrix(np.deg2rad(90), [0, 1, 0]) @ \
                     trimesh.transformations.rotation_matrix(np.deg2rad(90), [0, 0, 1])
    # Top
    poses['Top'] = trimesh.transformations.translation_matrix([0, 0, distance])

    # Bottom (Optional - might show the 'cut' surface)
    poses['Bottom'] = trimesh.transformations.translation_matrix([0, 0, -distance]) @ \
                      trimesh.transformations.rotation_matrix(np.deg2rad(180), [0, 1, 0])
    
    return poses

def process_and_render_with_orientation(stl_dir, output_dir, z_cutoff, orientation_deg=0):
    os.makedirs(output_dir, exist_ok=True)
    stl_files = sorted(glob(os.path.join(stl_dir, "*.stl")))
    
    # Offscreen renderer
    renderer = pyrender.OffscreenRenderer(224, 224)

    for stl_path in tqdm(stl_files):
        shape_id = os.path.basename(stl_path).replace('.stl', '')
        mesh = trimesh.load(stl_path)
        
        # 0. Apply the robot's orientation (Rotation around Z-axis/Stem)
        angle_rad = np.deg2rad(orientation_deg)
        rot_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
        mesh.apply_transform(rot_matrix)

        # 1. Slice and Center
        clean_mesh = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=[0,0,1], plane_origin=[0,0,z_cutoff], cap=True, engine='earcut'
        )
        clean_mesh.vertices -= clean_mesh.center_mass
        
        # 2. Setup Scene
        render_mesh = pyrender.Mesh.from_trimesh(clean_mesh)
        # Try lowering ambient_light from 0.4 to 0.2 if images are too bright
        scene = pyrender.Scene(ambient_light=[0.05, 0.05, 0.05], bg_color=[0, 0, 0])
        scene.add(render_mesh)
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        
        dist = clean_mesh.bounding_sphere.primitive.radius * 2.2
        poses = get_6view_poses(dist)

        for view_name, pose in poses.items():
            cam_node = scene.add(camera, pose=pose)
            light_node = scene.add(light, pose=pose)
            
            # --- RENDER BOTH COLOR AND DEPTH ---
            color, depth = renderer.render(scene)
            
            # Save Color Image
            img_color = Image.fromarray(color)
            img_color.save(os.path.join(output_dir, f"{shape_id}_{view_name}_rgb.png"))
            
            # --- IMPROVED DEPTH NORMALIZATION ---
            depth_map = depth.copy()
            mask = depth_map > 0
            
            if np.any(mask):
                # 1. Use percentiles to ignore outliers and focus on the object
                d_min = np.percentile(depth_map[mask], 1)  # Closest point
                d_max = np.percentile(depth_map[mask], 99) # Furthest point
                
                # 2. Clip and Normalize to 0-1
                depth_norm = np.clip((depth_map - d_min) / (d_max - d_min), 0, 1)
                
                # 3. Apply Gamma Correction to bring out details
                # A value > 1 (e.g., 1.5) darkens the mid-tones and adds contrast
                gamma = 0.75 
                depth_norm = np.power(depth_norm, gamma)
                
                # 4. Invert so closer is brighter (255) and further is darker
                depth_final = (1.0 - depth_norm) * 255
                depth_final[~mask] = 0  # Force background to absolute black
            else:
                depth_final = np.zeros_like(depth_map)

            img_depth = Image.fromarray(depth_final.astype(np.uint8))
            img_depth.save(os.path.join(output_dir, f"{shape_id}_{view_name}_depth.png"))

            scene.remove_node(cam_node)
            scene.remove_node(light_node)

    renderer.delete()


if __name__ == "__main__":
    process_and_render_with_orientation(STL_INPUT_DIR, RENDER_OUTPUT_DIR, Z_CUTOFF, orientation_deg=ORIENTATION)