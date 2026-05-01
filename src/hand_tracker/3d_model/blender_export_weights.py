import bpy
import json
import os

# Configuration
mesh_name = "Neo Hand Segmentation v1 decimated 10"  # Ensure this matches your mesh object name exactly

def export_hand_weights():
    # Detect the Desktop path automatically
    output_dir = "/home/yiting/Documents/GitHub/hand_tracking/configs"
    file_path = os.path.join(output_dir, "hand_skinning_weights.json")
    
    obj = bpy.data.objects.get(mesh_name)
    if not obj:
        print(f"ERROR: Object '{mesh_name}' not found. Please select your hand mesh.")
        return

    weight_map = {}
    group_names = {g.index: g.name for g in obj.vertex_groups}

    print(f"Processing {len(obj.data.vertices)} vertices...")

    for v in obj.data.vertices:
        v_weights = {}
        for g in v.groups:
            group_name = group_names.get(g.group)
            if group_name:
                # Store weight rounded to 4 decimal places
                v_weights[group_name] = round(g.weight, 4)
        
        weight_map[v.index] = v_weights

    try:
        with open(file_path, 'w') as f:
            json.dump(weight_map, f, indent=2)
        
        # Verify the file exists
        if os.path.exists(file_path):
            print("--------------------------------------------------")
            print(f"SUCCESS! Weights exported to: {file_path}")
            print(f"Total Vertices: {len(obj.data.vertices)}")
            print("--------------------------------------------------")
        else:
            print("ERROR: Script finished but file was not found on disk.")
            
    except Exception as e:
        print(f"PERMISSIONS ERROR: Could not write file. Try running Blender as Admin. Details: {e}")

export_hand_weights()