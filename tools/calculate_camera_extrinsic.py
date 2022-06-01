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

"""

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

import numpy as np
import argparse
from camera_pose_processing.CameraPoseCleaner import CameraPoseCleaner
from camera_pose_processing.CameraPoseSynchronizer import CameraPoseSynchronizer
from camera.calculate_extrinsic.CameraOptiExtrinsicCalculator import CameraOptiExtrinsicCalculator

def main():

    parser = argparse.ArgumentParser(description='Compute virtual optitrack camera to camera sensor extrinsic.')
    parser.add_argument('opti_poses', type=str, help="raw exported optitrack tracking data csv")
    parser.add_argument('frames', type=str, help='directory containing frames')
    parser.add_argument('--camera_intrinsic_matrix_file', type=str, default="camera/intrinsic.txt")
    parser.add_argument('--camera_distortion_coeff_file', type=str, default="camera/distortion.txt")
    parser.add_argument('--sync_frame_end', type=int, default=100)
    parser.add_argument('--output', type=str, default="camera/extrinsic.txt")

    args = parser.parse_args()

    camera_intrinsic_matrix = np.loadtxt(args.camera_intrinsic_matrix_file)
    camera_distortion_coeffs = np.loadtxt(args.camera_distortion_coeff_file)

    cam_pose_clean = CameraPoseCleaner()
    cam_pose_sync = CameraPoseSynchronizer()
    cam_opti_extr_calc = CameraOptiExtrinsicCalculator(camera_intrinsic_matrix, camera_distortion_coeffs)

    cleaned_opti_poses = cam_pose_clean.clean_camera_pose_file(args.opti_poses)
    pose_synchronization = cam_pose_sync.synchronize_camera_poses_and_frames(args.frames, cleaned_opti_poses, args.sync_frame_end)
    extrinsic = cam_opti_extr_calc.calculate_extrinsic(args.frames, cleaned_opti_poses, pose_synchronization)

    print("computed extrinsic:", extrinsic)
    
    if args.output:
        print("saving extrinsic to file {0}".format(args.output))
        np.savetxt(extrinsic, args.output)

if __name__ == "__main__":
    main()