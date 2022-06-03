"""
Uses MetadataGenerator and SemanticLabelingGenerator to generate scene labeling:
Specifically generates 00000_label.png and 00000_meta.json.
"""

import argparse
import yaml
import numpy as np

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from scene_labeling_generation.SemanticLabelingGenerator import SemanticLabelingGenerator
from scene_labeling_generation.MetadataGenerator import MetadataGenerator
from utils.object_utils import load_object_meshes

def main():
    parser = argparse.ArgumentParser(description='Generate semantic labeling and meta labeling')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')
    parser.add_argument('--camera_intrinsic_matrix_file', type=str, default="camera/intrinsic.txt")
    parser.add_argument('--camera_distortion_coeff_file', type=str, default="camera/distortion.txt")
    parser.add_argument('--camera_extrinsic_file', type=str, default="camera/extrinsic.txt")

    args = parser.parse_args()

    camera_intrinsic_matrix = np.loadtxt(args.camera_intrinsic_matrix_file)
    camera_distortion_coeffs = np.loadtxt(args.camera_distortion_coeff_file)
    camera_extrinsic = np.loadtxt(args.camera_extrinsic_file)

    scene_metadata_file = os.path.join(args.scene_dir, "scene_meta.yaml")
    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)
    
    objects = load_object_meshes(scene_metadata["objects"])

    frames_dir = os.path.join(args.scene_dir, "data")

    annotated_poses_csv = os.path.join(args.scene_dir, "annotated_object_poses", "annotated_object_poses.yaml")
    with open(annotated_poses_csv, "r") as file:
        annotated_poses_data = yaml.safe_load(file)

    annotated_poses_frameid = annotated_poses_data["frame"]
    annotated_poses = annotated_poses_data["object_poses"]
    annotated_poses = {k: np.array(v) for k, v in annotated_poses.items()}

    #TEST DATA.
    synchronized_poses = {0: np.eye(4), 1: np.eye(4)}
    synchronized_poses[1][:3,3] += np.array([0.3, 0, 0])

    #generate labels
    semantic_labeling_generator = SemanticLabelingGenerator(objects, camera_intrinsic_matrix, camera_distortion_coeffs, camera_extrinsic)
    semantic_labeling_generator.generate_semantic_labels(frames_dir, annotated_poses_frameid, annotated_poses, synchronized_poses)

    #metadata labeling requires semantic labeling
    metadata_labeling_generator = MetadataGenerator(camera_extrinsic)
    metadata_labeling_generator.generate_metadata_labels(frames_dir, annotated_poses_frameid, annotated_poses, synchronized_poses)

if __name__ == "__main__":
    main()