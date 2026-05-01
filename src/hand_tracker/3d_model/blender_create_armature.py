import bpy
import json
import mathutils

# 1. Update this path to your uploaded JSON
json_path = "/home/yiting/Documents/GitHub/hand_tracking/configs/hand_keypoint_map.json"
Y_OFFSET = -3.5  # Move 3.5mm from dorsal skin to bone center

with open(json_path, 'r') as f:
    coords = json.load(f)


def create_complete_armature(data):
    # Calculate Wrist Root
    wr = mathutils.Vector(data["Wrist_R"])
    wu = mathutils.Vector(data["Wrist_U"])
    wrist_root = (wr + wu) / 2
    wrist_root.y += Y_OFFSET

    # Create Armature
    arm_data = bpy.data.armatures.new("MonkeyHand_RigData")
    arm_obj = bpy.data.objects.new("MonkeyHand_Armature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Define hierarchies (Chain Name: [Joint List])
    # The first joint in each list will be parented to the Wrist_Root
    finger_chains = {
        "Thumb": ["Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip"],
        "Index": ["Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip"],
        "Middle": ["Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip"],
        "Ring": ["Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip"],
        "Small": ["Small_MCP", "Small_PIP", "Small_DIP", "Small_Tip"]
    }

    # Iterate through fingers
    for name, joints in finger_chains.items():
        parent_bone = None
        
        # A. Create the Metacarpal (Palm) bone for this finger
        # This bone goes from Wrist_Root to the first joint in the chain
        meta_name = f"{name}_Metacarpal"
        meta_bone = arm_data.edit_bones.new(meta_name)
        meta_bone.head = wrist_root
        
        target_pos = mathutils.Vector(data[joints[0]])
        target_pos.y += Y_OFFSET
        meta_bone.tail = target_pos
        parent_bone = meta_bone

        # B. Create the rest of the finger segments
        for i in range(len(joints) - 1):
            bone_name = joints[i]
            next_joint = joints[i+1]
            
            bone = arm_data.edit_bones.new(bone_name)
            
            head_pos = mathutils.Vector(data[bone_name])
            tail_pos = mathutils.Vector(data[next_joint])
            head_pos.y += Y_OFFSET
            tail_pos.y += Y_OFFSET
            
            bone.head = head_pos
            bone.tail = tail_pos
            
            if parent_bone:
                bone.parent = parent_bone
                # Note: Keep meta_bone disconnected to Wrist_Root 
                # but fingers connected to Meta_bones for better curling
                if "Metacarpal" not in parent_bone.name:
                    bone.use_connect = True
            
            parent_bone = bone

    bpy.ops.object.mode_set(mode='OBJECT')
    arm_obj.show_in_front = True

create_complete_armature(coords)