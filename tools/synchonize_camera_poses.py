"""

Generate synchronized_camera_poses using CameraPoseSynchronizer.
Required to run clean_camera_poses first to generate camera_pose_cleaned.csv

"""

import argparse

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from camera_pose_processing.CameraPoseCleaner import CameraPoseCleaner
from camera_pose_processing.CameraPoseSynchronizer import CameraPoseSynchronizer

def main():
    parser = argparse.ArgumentParser(description='Synchronize optitrack poses with frames')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')
    parser.add_argument('--sync_frame_end', type=int, default=100)
    args = parser.parse_args()

    frames_dir = os.path.join(args.scene_dir, "data")
    cleaned_poses_csv = os.path.join(args.scene_dir, "camera_poses", "camera_poses_cleaned.csv")

    cam_pose_cleaner = CameraPoseCleaner()
    cleaned_poses = cam_pose_cleaner.load_from_file(cleaned_poses_csv)

    cam_pose_synchronizer = CameraPoseSynchronizer()
    cam_pose_synchronizer.synchronize_camera_poses_and_frames(frames_dir, cleaned_poses, args.sync_frame_end, write_to_file=True)

if __name__ == "__main__":
    main()
