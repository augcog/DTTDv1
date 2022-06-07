"""
Clean the camera poses in a scene.
"""

import argparse

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from camera_pose_processing.CameraPoseCleaner import CameraPoseCleaner

def main():
    parser = argparse.ArgumentParser(description='Clean raw optitrack output.')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')
    
    args = parser.parse_args()

    cam_pose_cleaner = CameraPoseCleaner()
    raw_opti_poses = os.path.join(args.scene_dir, "camera_poses", "camera_poses.csv")
    cam_pose_cleaner.clean_camera_pose_file(raw_opti_poses, write_cleaned_to_file=True)


if __name__ == "__main__":
    main()
    