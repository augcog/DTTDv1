"""
Refine camera pose using ICP with objects.
Objects using manual pose annotation.
"""

import numpy as np
import open3d as o3d
import os

class ScenePoseRefiner():
    def __init__(self):
        pass

    @staticmethod
    def refine_pose(obj_pclds_in_sensor_coords, camera_pcld):
        obj_points = []
        for obj_pcld in obj_pclds_in_sensor_coords:
            obj_points.append(np.array(obj_pcld.points))

        obj_points = np.array(obj_points)
        obj_points = obj_points.reshape((-1, 3))

        all_object_pcld = o3d.geometry.PointCloud()
        all_object_pcld.points = o3d.utility.Vector3dVector(obj_points)

        trans_init = np.eye(4)

        criteria = o3d.registration.ICPConvergenceCriteria()
        criteria.max_iteration = 4

        pose_correction = o3d.registration.registration_icp(
            all_object_pcld, camera_pcld, .005, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            criteria)

        pose_correction = pose_correction.transformation
        return pose_correction

    @staticmethod
    def save_refined_poses_df(scene_dir, df):
        refined_path = os.path.join(scene_dir, "camera_poses", "camera_poses_synchronized_refined.csv")
        df.to_csv(refined_path)