"""
Manually annotate the first frame of a sequence of frames.
Outputs to file the annotated poses.
The output will be the object poses in the coordinate system of the camera sensor of the frame provided in frame.
"""

import argparse
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from data_processing.CameraPoseSynchronizer import CameraPoseSynchronizer
from manual_pose_annotation.ManualPoseAnnotator import ManualPoseAnnotator
from utils.object_utils import load_object_meshes
from utils.pose_dataframe_utils import convert_pose_df_to_dict

def main():

    parser = argparse.ArgumentParser(description='Compute virtual optitrack camera to camera sensor extrinsic.')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames))')
    parser.add_argument('--frame', type=int, default=0, help='which frame to use as coordinate system for annotation')

    args = parser.parse_args()

    scene_metadata_file = os.path.join(args.scene_dir, "scene_meta.yaml")
    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)
    
    objects = load_object_meshes(scene_metadata["objects"])

    cam_pose_sync = CameraPoseSynchronizer()
    synchronized_poses_csv = os.path.join(args.scene_dir, "camera_poses", "camera_poses_synchronized.csv")
    synchronized_poses = cam_pose_sync.load_from_file(synchronized_poses_csv)
    synchronized_poses = convert_pose_df_to_dict(synchronized_poses)

    manual_pose_annotator = ManualPoseAnnotator(objects)
    object_poses = manual_pose_annotator.annotate_pose(args.scene_dir, synchronized_poses, args.frame, ManualPoseAnnotator.icp_pose_initializer)
    
    object_poses_out = {}
    for obj_id, pose in object_poses.items():
        object_poses_out[obj_id] = pose.tolist()

    print("annotated object poses")
    print(object_poses_out) 

    #output to annotated_object_poses file

    annotated_object_poses_path = os.path.join(args.scene_dir, "annotated_object_poses")
    if not os.path.isdir(annotated_object_poses_path):
        os.mkdir(annotated_object_poses_path)

    output_file = os.path.join(annotated_object_poses_path, "annotated_object_poses.yaml")

    annotated_object_poses_out = {}
    annotated_object_poses_out["frame"] = args.frame
    annotated_object_poses_out["object_poses"] = object_poses_out

    with open(output_file, 'w') as outfile:
        yaml.dump(annotated_object_poses_out, outfile)

if __name__ == "__main__":
    main()
