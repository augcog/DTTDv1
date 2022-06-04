"""
Given objects and their poses in first camera frame, generate metadata label for every frame.
"""

"""
Given objects and their poses in first camera frame, generate semantic segmentation label for every frame.
"""

import numpy as np

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.affine_utils import invert_affine
from utils.frame_utils import write_meta, load_label

class MetadataGenerator():
    def __init__(self, virtual_to_sensor_extrinsic):
        self.camera_virtual_to_sensor_extrinsic = virtual_to_sensor_extrinsic

    def generate_metadata_labels(self, frames_dir, annotated_poses_single_frameid, annotated_poses_single_frame, synchronized_poses):

        sensor_to_virtual_extrinsic = invert_affine(self.camera_virtual_to_sensor_extrinsic)

        #apply extrinsic to convert every pose to actual camera sensor pose
        synchronized_poses_corrected = {}
        for frame_id, synchronized_pose in synchronized_poses.items():
            synchronized_poses_corrected[frame_id] = synchronized_pose @ sensor_to_virtual_extrinsic

        sensor_pose_annotated_frame = synchronized_poses_corrected[annotated_poses_single_frameid]
        sensor_pose_annotated_frame_inv = invert_affine(sensor_pose_annotated_frame)
        
        #use first frame coordinate system as world coordinates
        for frame_id, sensor_pose in synchronized_poses_corrected.items():

            sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ sensor_pose
            sensor_pose_in_annotated_coordinates_inv = invert_affine(sensor_pose_in_annotated_coordinates)
            
            label = load_label(frames_dir, frame_id)

            object_ids = np.unique(label)
            object_ids = object_ids[object_ids != 0]

            object_poses_in_sensor = {}

            for obj_id, annotated_obj_pose in annotated_poses_single_frame.items():
                if obj_id not in object_ids:
                    continue
                obj_pose_in_sensor = sensor_pose_in_annotated_coordinates_inv @ annotated_obj_pose
                object_poses_in_sensor[obj_id] = obj_pose_in_sensor.tolist()

            frame_metadata = {}
            frame_metadata["objects"] = object_ids.tolist()
            frame_metadata["object_poses"] = object_poses_in_sensor

            write_meta(frames_dir, frame_id, frame_metadata)





