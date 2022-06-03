"""
Manually annotate the first frame of a sequence of frames.
Outputs to file the annotated pose, doesn't have any dependency on optitrack poses whatsoever.
The output will be the object poses in the coordinate system of the camera sensor of the frame provided in first_frame_id.
"""

from PIL import Image
import numpy as np
import argparse
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from manual_pose_annotation.ManualPoseAnnotator import ManualPoseAnnotator
from utils.object_utils import load_object_meshes
from utils.frame_utils import load_rgb, load_depth

def main():

    parser = argparse.ArgumentParser(description='Compute virtual optitrack camera to camera sensor extrinsic.')
    parser.add_argument('first_frame_id', type=int, help='first frame that will be used to pose annotate')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames))')
    parser.add_argument('--camera_intrinsic_matrix_file', type=str, default="camera/intrinsic.txt")
    parser.add_argument('--camera_distortion_coeff_file', type=str, default="camera/distortion.txt")

    args = parser.parse_args()

    camera_intrinsic_matrix = np.loadtxt(args.camera_intrinsic_matrix_file)
    camera_distortion_coeffs = np.loadtxt(args.camera_distortion_coeff_file)

    scene_metadata_file = os.path.join(args.scene_dir, "scene_meta.yaml")
    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)
    
    objects = load_object_meshes(scene_metadata["objects"])

    frames_dir = os.path.join(args.scene_dir, "data")

    rgb_frame = load_rgb(frames_dir, args.first_frame_id)
    depth_frame = load_depth(frames_dir, args.first_frame_id)

    meta = os.path.join(args.scene_dir, "scene_meta.yaml")
    with open(meta, 'r') as file:
        meta = yaml.safe_load(file)
        cam_scale = meta["cam_scale"]

    manual_pose_annotator = ManualPoseAnnotator(objects, camera_intrinsic_matrix, camera_distortion_coeffs)
    object_poses = manual_pose_annotator.annotate_pose(rgb_frame, depth_frame, cam_scale, ManualPoseAnnotator.icp_pose_initializer)

    #output to annotated_object_poses file

    output_file = os.path.join(args.scnee_dir, "annotated_object_poses", "annotated_object_poses.yaml")

    annotated_object_poses_out = {}
    annotated_object_poses_out["frame"] = args.first_frame_id
    annotated_object_poses_out["object_poses"] = object_poses

    yaml.dump(annotated_object_poses_out, output_file)

if __name__ == "__main__":
    main()