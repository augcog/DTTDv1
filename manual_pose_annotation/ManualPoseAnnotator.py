"""
Goal: Write a manual pose annotator for the dataset.

Can initialize the pose using 
1) algorithm
2) ICP
3) idk other stuff

"""

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from manual_pose_annotation.depth_utils import fill_missing

class ManualPoseAnnotator:

    """
    objects contains triangle meshes for each object
    object coordinates are in meters
    """
    def __init__(self, objects, camera_intrinsic_matrix, camera_distortion_coefficients):
        self.objects = objects
        self.camera_intrinsic_matrix = camera_intrinsic_matrix
        self.camera_distortion_coefficients = camera_distortion_coefficients
        
    @staticmethod
    def pointcloud_from_rgb_depth(rgb, depth, depth_scale, intrinsic, distortion):

        points = np.array([[(k, i) for k in range(depth.shape[1])] for i in range(depth.shape[0])]).reshape((-1, 2)).astype(np.float32)
        Z = depth.flatten() * depth_scale

        f_x = intrinsic[0, 0]
        f_y = intrinsic[1, 1]
        c_x = intrinsic[0, 2]
        c_y = intrinsic[1, 2]

        points = points[Z > 0]
        Z = Z[Z > 0]

        # Step 1. Undistort.
        points_undistorted = cv2.undistortPoints(np.expand_dims(points, 1), intrinsic, distortion, P=intrinsic)
        points_undistorted = np.squeeze(points_undistorted, axis=1)

        # Step 2. Reproject.
        pts_xyz = []
        for idx in range(points_undistorted.shape[0]):
            z = Z[idx]
            x = (points_undistorted[idx, 0] - c_x) / f_x * z
            y = (points_undistorted[idx, 1] - c_y) / f_y * z
            pts_xyz.append([x, y, z])

        pts_xyz = np.array(pts_xyz)
        rgb = rgb.reshape((-1, 3))

        pcld = o3d.geometry.PointCloud()
        pcld.points = o3d.utility.Vector3dVector(pts_xyz)
        pcld.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32) / 255.)

        return pcld


    """
    icp pose initializer includes several steps:
    1) Prompt user to click on the RGB image wherever the center of the objects are
    2) Use Open3D's colored ICP to do some more fine-tuning
    """
    @staticmethod
    def icp_pose_initializer(rgb, depth, depth_scale, camera_intrinsic_matrix, camera_distortion_coefficients, objects):
        
        depth_smoothed = fill_missing(depth, 1 / depth_scale, 1)

        camera_pcld = ManualPoseAnnotator.pointcloud_from_rgb_depth(rgb, depth, depth_scale, camera_intrinsic_matrix, camera_distortion_coefficients)
        smoothed_camera_pcld = ManualPoseAnnotator.pointcloud_from_rgb_depth(rgb, depth_smoothed, depth_scale, camera_intrinsic_matrix, camera_distortion_coefficients)

        print('Please select the approximate center of each object in the scene')

        object_centers = {}

        for obj_id, obj_data in objects.items():
            print("Select the center for object named {0}".format(obj_data["name"]))

            plt.imshow(rgb)
            x, y = plt.ginput(1)[0]
            plt.close()

            object_centers[obj_id] = np.array(smoothed_camera_pcld.points)[int(y * depth.shape[1] + x)]

        object_pose_initializations = {}

        for obj_id in object_centers.keys():
            obj_mesh = objects[obj_id]["mesh"]
            obj_center = object_centers[obj_id]

            obj_mesh_with_normals = obj_mesh.compute_vertex_normals()

            target_vertices = obj_mesh_with_normals.vertices
            target_normals = obj_mesh_with_normals.vertex_normals

            target_pcld = o3d.geometry.PointCloud()
            target_pcld.points = target_vertices
            target_pcld.normals = target_normals

            trans_init = np.eye(4)
            trans_init[:3,3] = obj_center

            transform_obj_to_scene = o3d.registration.registration_icp(
                target_pcld, camera_pcld, .05, trans_init,
                o3d.registration.TransformationEstimationPointToPoint())

            print("icp result for obj {0}".format(obj_id))
            print(transform_obj_to_scene)

            transform_obj_to_scene = transform_obj_to_scene.transformation

            object_pose_initializations[obj_id] = transform_obj_to_scene

        return object_pose_initializations

    """
    rgb and depth are images
    depth_scale is the scale of the depth image, we multiply depth * depth_scale to get depth in meters
    initialization method is a function that takes in rgb, depth, depth scale, camera intrinsic matrix, 
    camera distortion coefficients, and list of object triangle meshes
    and returns a dict {obj_id: transform initialization}, the affine transform for each object
    """
    def annotate_pose(self, rgb, depth, depth_scale, initialization_method=None):
        if initialization_method:
            initial_poses = initialization_method(rgb, depth, depth_scale, self.camera_intrinsic_matrix, self.camera_distortion_coefficients, self.objects)
        else:
            initial_poses = {}
            for obj_id in self.objects.keys():
                initial_poses[obj_id] = np.eye(4)

        #TODO: Pose Annotator here
        annotated_poses = initial_poses

        return annotated_poses
