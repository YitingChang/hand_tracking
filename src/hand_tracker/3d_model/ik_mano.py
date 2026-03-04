from pathlib import Path
import os
import torch
import smplx
import numpy as np
import trimesh
from PIL import Image
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

MODEL_PATH = Path('/media/yiting/NewVolume/Analysis/hand_analysis/hand_models/models')
FRAME_NUMBER = 300

def fit_mano_to_keypoints(target_keypoints, model_path, iterations=3000):
    # 1. Initialize MANO Layer
    mano = smplx.create(model_path=str(model_path), 
                        model_type='mano', 
                        is_rhand=True, 
                        use_pca=False,
                        flat_hand_mean=True)
    
    # 2. Setup Parameters
    betas = torch.zeros([1, 10], requires_grad=True)
    with torch.no_grad():
        betas[0, 1] = -2.0  # Adjusting the "fullness" parameter to be thin

    global_orient = torch.zeros([1, 3], requires_grad=True)
    hand_pose = torch.zeros([1, 45], requires_grad=True) 
    translator = torch.zeros([1, 3], requires_grad=True)
    
    optimizer = torch.optim.Adam([betas, global_orient, hand_pose, translator], lr=0.01)
    
    # !!! CRITICAL: Convert mm to meters if necessary !!!
    if np.max(target_keypoints) > 10.0: # Simple check if data is in mm
        target_keypoints = target_keypoints / 1000.0
        print("Scaling input keypoints from mm to meters...")

    target_tensor = torch.tensor(target_keypoints, dtype=torch.float32)

    # Define MANO Tip Vertex Indices (for the 778-vertex mesh)
    # Order: Index, Middle, Small, Ring, Thumb (to match your 16-20 mapping)
    tip_vertex_indices = [320, 443, 671, 554, 744]

    # 3. Optimization Loop
    for i in range(iterations):
        optimizer.zero_grad()
        # Forward pass
        output = mano(betas=betas, 
                      global_orient=global_orient, 
                      hand_pose=hand_pose, 
                      transl=translator,
                      return_joints=True,
                      return_verts=True) 
        
        
        # 1. Get the 16 standard joints
        joints16 = output.joints[0] # (16, 3)
        
        # 2. Extract the 5 tips from the vertices
        # output.vertices shape is (1, 778, 3)
        tips = output.vertices[0, tip_vertex_indices] # (5, 3)
        
        # 3. Concatenate to get 21 points
        # pred_joints will now be (21, 3)
        pred_joints = torch.cat([joints16, tips], dim=0) 
        
        # 4. Compute Loss
        # Coordinate match
        loss = torch.mean((pred_joints - target_tensor)**2)
        
        # Shape Prior: Penalize 'fatness' (positive values of beta[1])
        # This keeps the hand from becoming too thick during optimization
        loss += 0.005 * torch.sum(betas**2) 
        loss += 0.01 * torch.relu(betas[0, 1]) # Extra penalty if beta[1] > 0

        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.6f}")

    final_output = mano(betas=betas, 
                        global_orient=global_orient, 
                        hand_pose=hand_pose, 
                        transl=translator,
                        return_verts=True)
    
    return final_output.vertices[0].detach().numpy(), mano.faces


def export_mesh_gif(mesh, output_path='hand_rotation.gif', num_frames=60):
    """
    Headless-safe GIF exporter using Matplotlib to render mesh triangles.
    """
    images = []
    # Extract vertices and faces from the trimesh object
    verts = mesh.vertices
    faces = mesh.faces

    for i in range(num_frames):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a collection of triangles
        # Note: We can subset faces (e.g., [::2]) if it's too slow to render
        poly = art3d.Poly3DCollection(verts[faces], alpha=0.7)
        poly.set_edgecolor('k')
        poly.set_linewidth(0.1)
        poly.set_facecolor('bisque') # Skin-like color
        ax.add_collection3d(poly)
        
        # Set view and labels
        ax.view_init(elev=20, azim=i * (360/num_frames))
        
        # Set axis limits based on mesh bounds
        v_min, v_max = verts.min(axis=0), verts.max(axis=0)
        center = (v_min + v_max) / 2
        max_range = (v_max - v_min).max() / 2
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.axis('off')
        
        # Save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        images.append(Image.open(buf))
        plt.close(fig)
        print(f"Rendering frame {i+1}/{num_frames}...", end='\r')

    # --- MODIFICATION TO SLOW DOWN GIF ---
    # duration=200 means each frame lasts 200ms (5 frames per second).
    # Since you have 60 frames, one full rotation will take 12 seconds.
    # Change to 150 for 9 seconds, or 300 for 18 seconds.
    images[0].save(
        output_path, 
        save_all=True, 
        append_images=images[1:], 
        duration=200, 
        loop=0
    )
    print(f"\nGIF saved to {output_path}")


def save_hand_mesh(mano_mesh, filename='reconstructed_hand.obj'):
    """
    Saves the MANO mesh to a 3D file.
    
    Parameters:
    - vertices: (778, 3) numpy array (from MANO output)
    - faces: (1538, 3) numpy array (from MANO model)
    - filename: Output path (supports .obj, .stl, .ply)
    """
    
    # Optional: Fix normals for better rendering
    mano_mesh.fix_normals()
    
    # Export
    mano_mesh.export(filename)
    print(f"Mesh successfully saved to: {filename}")


# --- Usage ---
mano_dir = Path('/media/yiting/NewVolume/Analysis/2025-08-19/mano')
trial_name = '2025-08-19_08-48-24'

# Load the traced keypoints (in meters)
traced_csv = mano_dir / f'mano_{trial_name}_frame{FRAME_NUMBER}.npy'
traced_points = np.load(traced_csv)

# Fit MANO model to the traced keypoints
vertices, faces = fit_mano_to_keypoints(traced_points, model_path=MODEL_PATH, iterations=6000)

# Create a mesh from the vertices and faces
hand_mesh = trimesh.Trimesh(vertices, faces)

# Save the hand mesh to an STL file
hand_mesh_path = mano_dir / f'hand_{trial_name}_frame{FRAME_NUMBER}.stl'
save_hand_mesh(hand_mesh, hand_mesh_path)

# Optional: Export a rotating GIF of the hand mesh
gif_path = mano_dir / f'hand_{trial_name}_frame{FRAME_NUMBER}.gif'
export_mesh_gif(hand_mesh, gif_path)
