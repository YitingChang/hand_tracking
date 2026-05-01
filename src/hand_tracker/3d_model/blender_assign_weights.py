import bpy
import bmesh

# Config: Adjust these based on your hand size (mm)
FINGER_RADIUS = 8.0   
PALM_RADIUS = 25.0    

mesh_obj = bpy.context.active_object
arm_obj = bpy.data.objects.get('MonkeyHand_Armature')

if not mesh_obj or not arm_obj:
    print("Error: Ensure the Hand Mesh is selected and Armature is named correctly.")
else:
    # 1. Collect bone data while in Object Mode
    bone_segments = []
    for bone in arm_obj.data.bones:
        head = arm_obj.matrix_world @ bone.head_local
        tail = arm_obj.matrix_world @ bone.tail_local
        radius = PALM_RADIUS if "Metacarpal" in bone.name else FINGER_RADIUS
        bone_segments.append({
            "name": bone.name,
            "head": head,
            "tail": tail,
            "vec": tail - head,
            "radius": radius
        })

    # 2. Get vertex coordinates using BMesh in Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh_obj.data)
    
    # Store world-space vertex positions and indices
    vert_data = []
    matrix_world = mesh_obj.matrix_world.copy()
    for v in bm.verts:
        vert_data.append((v.index, matrix_world @ v.co))
    
    # 3. Switch back to OBJECT mode to assign weights
    bpy.ops.object.mode_set(mode='OBJECT')

    # Iterate through bones and assign vertices
    for bone in bone_segments:
        # Get or create the vertex group
        vg = mesh_obj.vertex_groups.get(bone["name"]) or mesh_obj.vertex_groups.new(name=bone["name"])
        
        # Clear existing weights to prevent accumulation
        vg.remove(range(len(mesh_obj.data.vertices)))

        target_indices = []
        target_weights = []

        for v_idx, v_coord in vert_data:
            # Distance from point to line segment
            v_vec = v_coord - bone["head"]
            bone_vec = bone["vec"]
            t = max(0, min(1, v_vec.dot(bone_vec) / bone_vec.length_squared))
            dist = (v_coord - (bone["head"] + t * bone_vec)).length
            
            if dist < bone["radius"]:
                weight = 1.0 - (dist / bone["radius"])
                vg.add([v_idx], weight, 'REPLACE')

    print("Proximity Weighting Complete.")