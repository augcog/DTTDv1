"""

Clean camera poses using CameraPoseCleaner.
Generate synchronized_camera_poses using CameraPoseSynchronizer.
Fully generates scene directory:
1) data folder with rgb and depth (numbered 00000 - n)
2) scene_meta.yaml

"""

import argparse

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from data_processing.CameraPoseCleaner import CameraPoseCleaner
from data_processing.CameraPoseSynchronizer import CameraPoseSynchronizer

def main():
    parser = argparse.ArgumentParser(description='Synchronize optitrack poses with frames')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')
    args = parser.parse_args()

    cam_pose_cleaner = CameraPoseCleaner()
    cleaned_poses = cam_pose_cleaner.clean_camera_pose_file(args.scene_dir, write_cleaned_to_file=True)

    cam_pose_synchronizer = CameraPoseSynchronizer()
    cam_pose_synchronizer.synchronize_camera_poses_and_frames(args.scene_dir, cleaned_poses, write_to_file=True)

if __name__ == "__main__":
    main()
