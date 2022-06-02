"""
Manually annotate the first frame of a sequence of frames. Use the camera tracking to recover the pose in subsequent frames.
"""


from PIL import Image
import numpy as np
import argparse
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from camera_pose_processing.CameraPoseCleaner import CameraPoseCleaner
from camera_pose_processing.CameraPoseSynchronizer import CameraPoseSynchronizer
from manual_pose_annotation.ManualPoseAnnotator import ManualPoseAnnotator
from objects.object_utils import load_object_meshes

def main():

    parser = argparse.ArgumentParser(description='Compute virtual optitrack camera to camera sensor extrinsic.')
    parser.add_argument('opti_poses', type=str, help="raw exported optitrack tracking data csv")
    parser.add_argument('first_frame_id', type=int, help='first frame that will be used to pose annotate')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames))')
    parser.add_argument('object_ids', type=int, nargs='+', help='object ids of all objects in the scene we are annotating')
    parser.add_argument('--camera_intrinsic_matrix_file', type=str, default="camera/intrinsic.txt")
    parser.add_argument('--camera_distortion_coeff_file', type=str, default="camera/distortion.txt")
    parser.add_argument('--camera_extrinsic_file', type=str, default="camera/extrinsic.txt")
    parser.add_argument('--sync_frame_end', type=int, default=100)

    args = parser.parse_args()

    camera_intrinsic_matrix = np.loadtxt(args.camera_intrinsic_matrix_file)
    camera_distortion_coeffs = np.loadtxt(args.camera_distortion_coeff_file)
    
    objects = load_object_meshes(args.object_ids)

    frames_dir = os.path.join(args.scene_dir, "data")

    rgb_frame = np.array(Image.open(os.path.join(frames_dir, str(args.first_frame_id).zfill(5) + "_color.jpg")))
    depth_frame = np.array(Image.open(os.path.join(frames_dir, str(args.first_frame_id).zfill(5) + "_depth.png")))

    meta = os.path.join(args.scene_dir, "scene_meta.yaml")
    with open(meta, 'r') as file:
        meta = yaml.safe_load(file)
        cam_scale = meta["cam_scale"]

    manual_pose_annotator = ManualPoseAnnotator(objects, camera_intrinsic_matrix, camera_distortion_coeffs)
    object_poses = manual_pose_annotator.annotate_pose(rgb_frame, depth_frame, cam_scale, ManualPoseAnnotator.icp_pose_initializer)

    print("object poses")
    print(object_poses)
    exit()

    #visualize other frames

    extrinsic = np.loadtxt(args.camera_extrinsic_file)
    cam_pose_clean = CameraPoseCleaner()
    cam_pose_sync = CameraPoseSynchronizer()

    cleaned_opti_poses = cam_pose_clean.clean_camera_pose_file(args.opti_poses)
    pose_synchronization = cam_pose_sync.synchronize_camera_poses_and_frames(frames_dir, cleaned_opti_poses, args.sync_frame_end)


if __name__ == "__main__":
    main()