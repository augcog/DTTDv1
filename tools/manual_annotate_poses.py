"""
Manually annotate the first frame of a sequence of frames.
Outputs to file the annotated poses.
The output will be the object poses in the coordinate system of the camera sensor of the frame provided in frame.
"""

import argparse
from functools import partial
import numpy as np
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from data_processing import CameraPoseSynchronizer
from manual_pose_annotation import ManualPoseAnnotator
from utils.affine_utils import invert_affine
from utils.camera_utils import load_extrinsics
from utils.constants import SCENES_DIR
from utils.object_utils import load_object_meshes
from utils.pose_dataframe_utils import convert_pose_df_to_dict

def main():

    parser = argparse.ArgumentParser(description='Compute virtual optitrack camera to camera sensor extrinsic.')
    parser.add_argument('scene_name', type=str, help='scene directory (contains scene_meta.yaml and data (frames))')
    parser.add_argument('--frame', type=int, default=0, help='which frame to use as coordinate system for annotation')
    parser.add_argument('--use_prev', help='use previous annotation')
    parser.add_argument('--no-use-prev', dest='use_prev', action='store_false')
    parser.add_argument('--refresh-extrinsic', default=False, action="store_true")
    parser.add_argument('--transfer-annotations', type=str, help="transfer annotations from a different scene (should be the same object configuration)")
    parser.set_defaults(use_prev=True)
    args = parser.parse_args()

    scene_dir = os.path.join(SCENES_DIR, args.scene_name)

    scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")
    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)
    
    objects = load_object_meshes(scene_metadata["objects"])

    synchronized_poses_csv = os.path.join(scene_dir, "camera_poses", "camera_poses_synchronized.csv")
    synchronized_poses = CameraPoseSynchronizer().load_from_file(synchronized_poses_csv)
    synchronized_poses = convert_pose_df_to_dict(synchronized_poses)

    annotated_object_poses_path = os.path.join(scene_dir, "annotated_object_poses")
    if not os.path.isdir(annotated_object_poses_path):
        os.mkdir(annotated_object_poses_path)

    manual_pose_annotator = ManualPoseAnnotator(objects)

    annotation_file = os.path.join(annotated_object_poses_path, "annotated_object_poses.yaml")

    """
    Transfer annotations from a different scene
    """
    if args.transfer_annotations:

        transfer_annotation_scene_dir = os.path.join(SCENES_DIR, args.transfer_annotations)

        transfer_annotations_poses_path = os.path.join(transfer_annotation_scene_dir, "annotated_object_poses", "annotated_object_poses.yaml")
        with open(transfer_annotations_poses_path, "r") as file:
            transfer_annotations_data = yaml.safe_load(file)
        transfer_annotations_frame = transfer_annotations_data["frame"]

        # object -> sensor annot
        transfer_annotations_poses = transfer_annotations_data["object_poses"]
        transfer_annotations_poses = {k: np.array(v) for k, v in transfer_annotations_poses.items()}
        transfer_annotations_scene_meta_path = os.path.join(transfer_annotation_scene_dir, "scene_meta.yaml")
        with open(transfer_annotations_scene_meta_path, "r") as file:
            transfer_annotations_scene_meta = yaml.safe_load(file)
        transfer_annotations_camera_name = transfer_annotations_scene_meta["camera"]
        
        # virtual -> sensor annot
        transfer_annotations_extrinsic = load_extrinsics(transfer_annotations_camera_name, scene_dir=transfer_annotation_scene_dir, use_archive=True)

        transfer_annotations_synchronized_poses_path = os.path.join(transfer_annotation_scene_dir, "camera_poses", "camera_poses_synchronized.csv")
        transfer_annotations_synchronized_poses = CameraPoseSynchronizer().load_from_file(transfer_annotations_synchronized_poses_path)
        transfer_annotations_synchronized_poses = convert_pose_df_to_dict(transfer_annotations_synchronized_poses)

        # virtual -> opti
        transfer_annotations_synchronized_pose_annotated_frame = transfer_annotations_synchronized_poses[transfer_annotations_frame]

        # our virtual -> opti
        our_pose = synchronized_poses[args.frame]

        our_camera = scene_metadata["camera"]

        # our virtual -> our sensor
        our_extrinsic = load_extrinsics(our_camera, scene_dir, use_archive=True)

        transferred_annotation_poses = {}

        for obj_id, pose in transfer_annotations_poses.items():
            object_to_virtual = invert_affine(transfer_annotations_extrinsic) @ pose
            object_to_opti = transfer_annotations_synchronized_pose_annotated_frame @ object_to_virtual
            object_to_our_virtual = invert_affine(our_pose) @ object_to_opti
            object_to_our_sensor = our_extrinsic @ object_to_our_virtual
            transferred_annotation_poses[obj_id] = object_to_our_sensor

        def our_init(*args, **kwargs):
            return transferred_annotation_poses

        initializer = our_init

    elif args.use_prev and not args.frame and os.path.isfile(annotation_file):
        print("Using previous annotation.")

        #need to retrieve frameid from old annotation
        with open(annotation_file, "r") as file:
            annotated_poses_data = yaml.safe_load(file)
        args.frame = annotated_poses_data["frame"]

        initializer = partial(ManualPoseAnnotator.previous_initializer, scene_dir)
    else:
        initializer = ManualPoseAnnotator.icp_pose_initializer
    object_poses = manual_pose_annotator.annotate_pose(scene_dir, synchronized_poses, args.frame, initializer, use_archive_extrinsic=(not args.refresh_extrinsic))
    
    object_poses_out = {}
    for obj_id, pose in object_poses.items():
        object_poses_out[obj_id] = pose.tolist()

    print("annotated object poses")
    print(object_poses_out) 

    #output to annotated_object_poses file

    annotated_object_poses_out = {}
    annotated_object_poses_out["frame"] = args.frame
    annotated_object_poses_out["object_poses"] = object_poses_out

    with open(annotation_file, 'w') as outfile:
        yaml.dump(annotated_object_poses_out, outfile)

if __name__ == "__main__":
    main()
