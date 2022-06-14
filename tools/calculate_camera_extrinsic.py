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
import numpy as np

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from camera.calculate_extrinsic.CameraOptiExtrinsicCalculator import CameraOptiExtrinsicCalculator
from camera_pose_processing.CameraPoseSynchronizer import CameraPoseSynchronizer
from utils.pose_dataframe_utils import convert_pose_df_to_dict

def main():

    parser = argparse.ArgumentParser(description='Compute virtual optitrack camera to camera sensor extrinsic.')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')
    parser.add_argument('--camera_intrinsic_matrix_file', type=str, default="camera/intrinsic.txt")
    parser.add_argument('--camera_distortion_coeff_file', type=str, default="camera/distortion.txt")
    parser.add_argument('--output', type=str, default="camera/extrinsic.txt")

    args = parser.parse_args()

    camera_intrinsic_matrix = np.loadtxt(args.camera_intrinsic_matrix_file)
    camera_distortion_coeffs = np.loadtxt(args.camera_distortion_coeff_file)

    cam_opti_extr_calc = CameraOptiExtrinsicCalculator(camera_intrinsic_matrix, camera_distortion_coeffs)

    cam_pose_sync = CameraPoseSynchronizer()
    synchronized_poses_csv = os.path.join(args.scene_dir, "camera_poses", "synchronized_camera_poses.csv")
    synchronized_poses = cam_pose_sync.load_from_file(synchronized_poses_csv)
    synchronized_poses = convert_pose_df_to_dict(synchronized_poses)

    frames_dir = os.path.join(args.scene_dir, "data")
    
    extrinsic = cam_opti_extr_calc.calculate_extrinsic(frames_dir, synchronized_poses)

    print("computed extrinsic:", extrinsic)
    
    if args.output:
        print("saving extrinsic to file {0}".format(args.output))
        np.savetxt(args.output, extrinsic)

if __name__ == "__main__":
    main()
