from pathlib import Path
import os
import trimesh
import numpy as np
import pandas as pd

# The objects can be presented in different orientations.
# This script computes the similarity between two orientation conditions for each object.
# The symmetry score is based on the similarity of features extracted from different views of the same object under different orientations.

# --- CONFIGURATION ---
DATA_ROOT = Path("/media/yiting/NewVolume/Data")
SHAPE_DIR = DATA_ROOT / "Shapes"
STL_INPUT_DIR = SHAPE_DIR / "shapes_stl"
ORIENTATION = 180  # Degrees to rotate around Z-axis (stem).
RESOLUTION = 128  # Voxel grid resolution (e.g., 64x64x64)

def compute_orientation_similarity(stl_path, resolution=RESOLUTION):
    """
    Computes IoU similarity between 0 and ORIENTATION.
    Resolution defines the voxel grid size (e.g., 64x64x64).
    """
    try:
        # 1. Load the mesh
        mesh = trimesh.load(stl_path)
        
        # 2. Center the mesh at its bounding box centroid 
        # This ensures the rotation is 'in-place'
        mesh.vertices -= mesh.bounding_box.centroid
        
        # 3. Create the ORIENTATION-degree orientation
        mesh_2 = mesh.copy()
        rot_matrix = trimesh.transformations.rotation_matrix(np.pi * ORIENTATION / 180, [0, 0, 1])
        mesh_2.apply_transform(rot_matrix)
        
        # 4. Voxelize both
        # Pitch is the size of a single voxel; smaller = more precise but slower
        pitch = mesh.extents.max() / resolution
        vox_0 = mesh.voxelized(pitch=pitch).matrix
        vox_2 = mesh_2.voxelized(pitch=pitch).matrix
        
        # 5. Ensure the voxel grids are the same shape for comparison
        # We use the 'encoding' or 'matrix' overlap
        intersection = np.logical_and(vox_0, vox_2).sum()
        union = np.logical_or(vox_0, vox_2).sum()
        
        iou = intersection / union if union > 0 else 0
        return iou

    except Exception as e:
        print(f"Error processing {stl_path}: {e}")
        return None

# --- Main Execution ---
results = []
for filename in os.listdir(STL_INPUT_DIR):
    if filename.lower().endswith('.stl'):
        path = os.path.join(STL_INPUT_DIR, filename)
        print(f"Processing {filename}...")
        
        score = compute_orientation_similarity(path)
        results.append({'filename': filename, 'object_id': filename.split('.')[0], 'similarity_score': score})

# Save to CSV
df = pd.DataFrame(results)
save_path = SHAPE_DIR / f"symmetry_results_res-{RESOLUTION}_ori-{ORIENTATION}.csv"
df.to_csv(save_path, index=False)
print(f"Processing complete. Results saved to {save_path}")