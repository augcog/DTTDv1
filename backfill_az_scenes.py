import os
from utils.camera_utils import load_intrinsic_static, write_static_intrinsic

camera_name = "az_camera1"

camera_intr = load_intrinsic_static(camera_name)

for scene in os.listdir("scenes"):
    scene_dir = os.path.join("scenes", scene)
    if not os.path.isdir(scene_dir):
        continue
    write_static_intrinsic(camera_name, scene_dir, raw=False)
    write_static_intrinsic(camera_name, scene_dir, raw=True)

for scene in os.listdir("extrinsics_scenes"):
    scene_dir = os.path.join("scenes", scene)
    if not os.path.isdir(scene_dir):
        continue
    write_static_intrinsic(camera_name, scene_dir, raw=False)
    write_static_intrinsic(camera_name, scene_dir, raw=True)