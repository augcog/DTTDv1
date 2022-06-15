"""
Given objects and their poses in first camera frame, generate semantic segmentation label for every frame.

NOTE: This doesn't deal with scene occlusions, only object occlusions.
"""

from tokenize import Number
import cv2
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.affine_utils import invert_affine
from utils.camera_utils import load_extrinsics, load_intrinsics, load_distortion
from utils.frame_utils import load_rgb, write_debug_label, write_label, write_debug_rgb
from utils.mesh_utils import uniformly_sample_mesh_with_textures_as_colors

class SemanticLabelingGenerator():
    def __init__(self, objects, number_of_points=1000000):
        self._objects = {}
        for obj_id, obj_data in objects.items():
            obj_pcld = uniformly_sample_mesh_with_textures_as_colors(obj_data["mesh"], obj_data["texture"], number_of_points)
            self._objects[obj_id] = obj_pcld
        
        self.number_of_points = number_of_points

    def generate_semantic_labels(self, scene_dir, annotated_poses_single_frameid, annotated_poses_single_frame, synchronized_poses, debug=False):

        frames_dir = os.path.join(scene_dir, "data")

        scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")

        with open(scene_metadata_file, 'r') as file:
            scene_metadata = yaml.safe_load(file)

        camera_name = scene_metadata["camera"]

        camera_intrinsics = load_intrinsics(camera_name)
        camera_distortion = load_distortion(camera_name)
        camera_extrinsics = load_extrinsics(camera_name)

        sensor_to_virtual_extrinsic = invert_affine(camera_extrinsics)

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
            object_pcld_transformed[obj_id] = obj_pcld

        rgb = load_rgb(frames_dir, annotated_poses_single_frameid)
        h, w, _ = rgb.shape
        
        #use first frame coordinate system as world coordinates
        for frame_id, sensor_pose in synchronized_poses_corrected.items():

            sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ sensor_pose
            sensor_pose_in_annotated_coordinates_inv = invert_affine(sensor_pose_in_annotated_coordinates)
            sensor_rot = sensor_pose_in_annotated_coordinates[:3,:3]
            sensor_trans = sensor_pose_in_annotated_coordinates[:3,3]
            sensor_rvec = R.from_matrix(sensor_rot).as_rotvec()

            #fill these, then argmin over the depth
            label_out = np.zeros((h, w, len(object_pcld_transformed))).astype(np.uint16)
            depth_out = np.ones_like(label_out).astype(np.float32) * 10000 #inf depth

            if debug:
                rgb_out = np.zeros((h, w, 3, len(object_pcld_transformed))).astype(np.uint8)
                object_colors = {}
                for obj_id, obj_pcld in self._objects.items():
                    object_colors[obj_id] = (np.array(obj_pcld.colors) * 255).astype(np.uint8)

            for idx, (obj_id, obj_pcld) in enumerate(object_pcld_transformed.items()):

                obj_pts = np.array(obj_pcld.points)

                obj_pts_projected, _ = cv2.projectPoints(obj_pts, sensor_rvec, sensor_trans, camera_intrinsics, camera_distortion)
                obj_pts_projected = obj_pts_projected.squeeze(1) #(Nx2)
                obj_pts_projected = np.round(obj_pts_projected).astype(np.int)

                mask1 = obj_pts_projected[:,0] >= 0
                mask2 = obj_pts_projected[:,0] < w
                mask3 = obj_pts_projected[:,1] >= 0
                mask4 = obj_pts_projected[:,1] < h

                mask = mask1 * mask2 * mask3 * mask4

                obj_pcld = o3d.geometry.PointCloud(obj_pcld) #copy constructor

                obj_pcld = obj_pcld.transform(sensor_pose_in_annotated_coordinates_inv)
                
                
                obj_pts_in_sensor_coordinates = np.array(obj_pcld.points)

                #need z-coordinate for buffer
                obj_zs = obj_pts_in_sensor_coordinates[:,2] #(Nx1)

                obj_pts_projected = obj_pts_projected[mask]
                obj_zs = obj_zs[mask]

                obj_pts_projected_flattened = obj_pts_projected[:,0] + obj_pts_projected[:,1] * w

                label_buffer = label_out[:,:,idx].flatten()
                depth_buffer = depth_out[:,:,idx].flatten()

                if debug:
                    rgb_buffer = rgb_out[:,:,:,idx].reshape((-1, 3))
                
                label_buffer[obj_pts_projected_flattened] = obj_id
                depth_buffer[obj_pts_projected_flattened] = obj_zs

                if debug:
                    obj_pose_in_sensor = sensor_pose_in_annotated_coordinates_inv @ annotated_poses_single_frame[obj_id]
                    obj_rot_in_sensor = obj_pose_in_sensor[:3,:3]

                    obj_colors = object_colors[obj_id][mask]

                    normals = np.array(obj_pcld.normals)
                    normals = normals[mask]
                    normals = normals @ obj_rot_in_sensor.T

                    normals_mask = normals[:,2] < 0 #only get points with normals facing camera
                    
                    obj_colors = obj_colors[normals_mask]
                    normals = normals[normals_mask]
                    obj_pts_projected_flattened = obj_pts_projected_flattened[normals_mask]

                    rgb_buffer[obj_pts_projected_flattened] = obj_colors

                label_out[:,:,idx] = label_buffer.reshape((h, w))
                depth_out[:,:,idx] = depth_buffer.reshape((h, w))

                if debug:
                    rgb_out[:,:,:,idx] = rgb_buffer.reshape((h, w, 3))

            depth_argmin = np.expand_dims(np.argmin(depth_out, axis=-1), -1)
            label_out = np.take_along_axis(label_out, depth_argmin, axis=-1).squeeze(-1)

            if debug:
                rgb_debug = np.copy(rgb)

                depth_argmin = np.tile(np.expand_dims(depth_argmin, 2), (1, 1, 3, 1))
                rgb_out = np.take_along_axis(rgb_out, depth_argmin, axis=-1).squeeze(-1)
                
                rgb_debug = rgb_debug.reshape((-1, 3))
                rgb_out = rgb_out.reshape((-1, 3))

                rgb_out_mask = np.sum(rgb_out, axis=-1) > 0
                rgb_debug[rgb_out_mask] = rgb_out[rgb_out_mask]

                rgb_debug = rgb_debug.reshape((h, w, 3))

                write_debug_rgb(frames_dir, frame_id, rgb_debug)

            write_label(frames_dir, frame_id, label_out)

            if debug:
                write_debug_label(frames_dir, frame_id, label_out * 10000)
