bl_info = {
    "name": "Normal Repainter",
    "blender": (4, 3, 2),
    "category": "Image",
    "version": (1, 0),
    "author": "SYL",
    "description": "Repaints normals to follow base color paint strokes."
}

import bpy
import os
from . import processor

class NMSFProperties(bpy.types.PropertyGroup):
    base_color: bpy.props.StringProperty(
        name="Base Color Map",
        subtype='FILE_PATH'
    )
    normal_map: bpy.props.StringProperty(
        name="Normal Map",
        subtype='FILE_PATH'
    )
    output_path: bpy.props.StringProperty(
        name="Output Path",
        subtype='FILE_PATH',
        default="//output_normal_map.png"
    )

class NMSF_OT_Process(bpy.types.Operator):
    bl_idname = "nmsf.process"
    bl_label = "Process Normal Map"
    
    def execute(self, context):
        props = context.scene.nmsf_props
        base_color = bpy.path.abspath(props.base_color)
        normal_map = bpy.path.abspath(props.normal_map)
        output_path = bpy.path.abspath(props.output_path)
        
        if not (os.path.exists(base_color) and os.path.exists(normal_map)):
            self.report({'ERROR'}, "Please select valid image paths")
            return {'CANCELLED'}
        
        processor.process_auto(base_color, normal_map, output_path)
        self.report({'INFO'}, f"Processed normal map saved to {output_path}")
        return {'FINISHED'}

class NMSF_PT_Panel(bpy.types.Panel):
    bl_label = "Normal Map Seam Fixer"
    bl_idname = "NMSF_PT_panel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Seam Fixer'

    def draw(self, context):
        layout = self.layout
        props = context.scene.nmsf_props
        
        layout.prop(props, "base_color")
        layout.prop(props, "normal_map")
        layout.prop(props, "output_path")
        layout.operator("nmsf.process", icon='PLAY')

classes = (NMSFProperties, NMSF_OT_Process, NMSF_PT_Panel)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.nmsf_props = bpy.props.PointerProperty(type=NMSFProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.nmsf_props

if __name__ == "__main__":
    register()
