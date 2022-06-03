"""
Given objects and their poses in first camera frame, generate semantic segmentation label for every frame.
"""

from multiprocessing import synchronize
import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.affine_utils import invert_affine
from utils.frame_utils import load_rgb, load_depth
from utils.pointcloud_utils import pointcloud_from_rgb_depth

class SemanticLabelingGenerator():
    def __init__(self, objects, camera_intrinsic_matrix, camera_distortion_coefficients, virtual_to_sensor_extrinsic):
        self._objects = {}
        for obj_id, obj_data in objects.items():
            obj_pcld = obj_data["mesh"].sample_points_uniformly(number_of_points=10000)
            self._objects[obj_id] = obj_pcld
        self.camera_intrinsic_matrix = camera_intrinsic_matrix
        self.camera_distortion_coefficients = camera_distortion_coefficients
        self.camera_virtual_to_sensor_extrinsic = virtual_to_sensor_extrinsic

    def generate_semantic_labels(self, frames_dir, annotated_poses_single_frameid, annotated_poses_single_frame, synchronized_poses):

        sensor_to_virtual_extrinsic = invert_affine(self.camera_virtual_to_sensor_extrinsic)

        #apply extrinsic to convert every pose to actual camera sensor pose
        synchronized_poses_corrected = {}
        for frame_id, synchronized_pose in synchronized_poses.items():
            synchronized_poses_corrected[frame_id] = synchronized_pose @ sensor_to_virtual_extrinsic

        sensor_pose_annotated_frame = synchronized_poses_corrected[annotated_poses_single_frameid]
        sensor_pose_annotated_frame_inv = invert_affine(sensor_pose_annotated_frame)

        object_pcld_transformed = {}
        for obj_id, obj_pcld in self._objects.items():
            annotated_obj_pose = annotated_poses_single_frame[obj_id]
            obj_pcld = obj_pcld.transform(annotated_obj_pose)
            object_pcld_transformed[obj_id] = np.array(obj_pcld.points)
        
        #use first frame coordinate system as world coordinates
        for frame_id, sensor_pose in synchronized_poses_corrected.items():

            sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ sensor_pose_annotated_frame
            sensor_pose_in_annotated_coordinates_inv = invert_affine(sensor_pose_in_annotated_coordinates)
            sensor_rot = sensor_pose_in_annotated_coordinates[:3,:3]
            sensor_trans = sensor_pose_in_annotated_coordinates[:3,3]
            sensor_rvec = R.from_matrix(sensor_rot).as_rotvec()

            for obj_id, obj_pcld in object_pcld_transformed.items():

                obj_pts = np.array(obj_pcld.points)

                obj_pts_projected, _ = cv2.projectPoints(obj_pts, sensor_rvec, sensor_trans, self.camera_intrinsic_matrix, self.camera_distortion_coefficients)
                print(obj_pts_projected.shape)

                obj_pcld = o3d.geometry.PointCloud(obj_pcld) #copy constructor
                obj_pcld = obj_pcld.transform(sensor_pose_in_annotated_coordinates_inv)
                obj_pts_in_sensor_coordinates = np.array(obj_pcld.points)

                #need z-coordinate for buffer
                obj_zs = obj_pts[:,2]


