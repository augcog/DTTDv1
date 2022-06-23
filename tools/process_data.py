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

from data_processing import CameraPoseCleaner
from data_processing import CameraPoseSynchronizer
from pose_refinement import OptiKFSmoother
from utils.constants import EXTRINSICS_DIR, SCENES_DIR
from utils.datetime_utils import get_latest_str_from_str_time_list

def main():
    parser = argparse.ArgumentParser(description='Synchronize optitrack poses with frames')
    parser.add_argument('--scene_name', type=str, help='name of scene')
    parser.add_argument('--extrinsic', default=False, action="store_true", help='processing a extrinsic scene')
    parser.add_argument('--rewrite_images', action='store_true')
    parser.add_argument('--no-rewrite-images', dest='rewrite_images', action='store_false')
    parser.set_defaults(rewrite_images=True)
    args = parser.parse_args()

    if not args.scene_name and not args.extrinsic:
        print("Must be a scene capture (indicate a scene_name) or an extrinsic capture (use --extrinsic flag).")
        exit(-1)

    if args.extrinsic and args.scene_name:
        scene_dir = os.path.join(EXTRINSICS_DIR, args.scene_name)
    elif args.extrinsic:
        extrinsic_scenes = list(os.listdir(EXTRINSICS_DIR))
        latest_extrinsic_scene = get_latest_str_from_str_time_list(extrinsic_scenes)

        print("using extrinsic scene {0}".format(latest_extrinsic_scene))
        
        scene_dir = os.path.join(EXTRINSICS_DIR, latest_extrinsic_scene)
    else:
        scene_dir = os.path.join(SCENES_DIR, args.scene_name)

    cam_pose_cleaner = CameraPoseCleaner()
    cleaned_poses = cam_pose_cleaner.clean_camera_pose_file(scene_dir, write_cleaned_to_file=True)

    print("Poses cleaned!")

    cam_pose_smoother = OptiKFSmoother()
    smoothed_poses = cam_pose_smoother.smooth_opti_poses_kf(scene_dir, cleaned_poses, write_smoothed_to_file=True)

    print("Poses smoothed!")

    cam_pose_synchronizer = CameraPoseSynchronizer()
    cam_pose_synchronizer.synchronize_camera_poses_and_frames(scene_dir, smoothed_poses, write_to_file=True, rewrite_images = args.rewrite_images)

    print("Poses synchronized!")

if __name__ == "__main__":
    main()
