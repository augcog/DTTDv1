"""
Need to compute the extrinsic from OptiTrack virtual marker center of mass to camera sensor.

Procedure:
1) Record a sequence of frames that include an ARUCO marker placed at or with known transform from OptiTrack origin.
2) Compute ARUCO marker pose in camera sensor coordinate system
3) Synchronize OptiTrack poses with camera frames using CameraPoseSynchronizer
4) Solve for extrinsic using OptiTrack -> Camera_sensor and Camera_virtual -> OptiTrack transforms

This tool assumes you have recorded data:
1) OptiTrack poses (raw)
2) RGB frames (named 0.png, 1.png, etc.)

There must have been a synchronization phase at the start of the sequence (shake method :))

This code assumes you have run clean_camera_poses and synchronize_camera_poses already

"""

import argparse

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from calculate_extrinsic import CameraOptiExtrinsicCalculator
from data_processing import CameraPoseSynchronizer
from utils.constants import EXTRINSICS_DIR
from utils.datetime_utils import get_latest_str_from_str_time_list
from utils.pose_dataframe_utils import convert_pose_df_to_dict

def main():

    parser = argparse.ArgumentParser(description='Compute virtual optitrack camera to camera sensor extrinsic.')
    parser.add_argument('--scene_name', default='', type=str, help='Which scene to use to calculate extrinsic. Else, uses latest extrinsic scene in extrinsics dir')

    args = parser.parse_args()

    cam_opti_extr_calc = CameraOptiExtrinsicCalculator()

    if args.scene_name:
        scene_dir = os.path.join(EXTRINSICS_DIR, args.scene_name)
    else:
        extrinsic_scenes = list(os.listdir(EXTRINSICS_DIR))
        latest_extrinsic_scene = get_latest_str_from_str_time_list(extrinsic_scenes)

        print("using extrinsic scene {0}".format(latest_extrinsic_scene))

        scene_dir = os.path.join(EXTRINSICS_DIR, latest_extrinsic_scene)

    cam_pose_sync = CameraPoseSynchronizer()
    synchronized_poses_csv = os.path.join(scene_dir, "camera_poses", "camera_poses_synchronized.csv")
    synchronized_poses = cam_pose_sync.load_from_file(synchronized_poses_csv)
    synchronized_poses = convert_pose_df_to_dict(synchronized_poses)
    
    extrinsic = cam_opti_extr_calc.calculate_extrinsic(scene_dir, synchronized_poses, write_to_file=True)
    
    print("computed extrinsic:", extrinsic)
    
if __name__ == "__main__":
    main()
