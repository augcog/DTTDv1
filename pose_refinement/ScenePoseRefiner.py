"""
Refine camera pose using manual pose annotation.
"""

import cv2
from functools import partial
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import yaml

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.affine_utils import invert_affine
from utils.camera_utils import load_distortion, load_extrinsics, load_intrinsics
from utils.frame_utils import load_depth, load_rgb, load_bgr
from utils.pointcloud_utils import pointcloud_from_rgb_depth
from utils.pose_dataframe_utils import convert_pose_dict_to_df

class ScenePoseRefiner():
    def __init__(self, objects={}, number_of_points=10000):
        self._objects = {}
        for obj_id, obj_data in objects.items():
            obj_pcld = obj_data["mesh"].sample_points_uniformly(number_of_points=number_of_points)
            self._objects[obj_id] = {"pcld": obj_pcld, "mesh": o3d.geometry.TriangleMesh(obj_data["mesh"])}
        
        self.number_of_points = number_of_points

    @staticmethod
    def refine_pose_icp(obj_pclds_in_sensor_coords, camera_pcld):
        obj_points = []
        for obj_pcld in obj_pclds_in_sensor_coords:
            obj_points.append(np.array(obj_pcld.points))

        obj_points = np.array(obj_points)
        obj_points = obj_points.reshape((-1, 3))

        all_object_pcld = o3d.geometry.PointCloud()
        all_object_pcld.points = o3d.utility.Vector3dVector(obj_points)

        trans_init = np.eye(4)

        criteria = o3d.registration.ICPConvergenceCriteria()
        criteria.max_iteration = 8

        pose_correction = o3d.registration.registration_icp(
            all_object_pcld, camera_pcld, .015, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            criteria)

        pose_correction = pose_correction.transformation
        return pose_correction

    @staticmethod
    def save_refined_poses_df(scene_dir, df):
        refined_path = os.path.join(scene_dir, "camera_poses", "camera_poses_synchronized_refined.csv")
        df.to_csv(refined_path)

    def refine_poses(self, scene_dir, annotated_poses_single_frameid, annotated_poses_single_frame, synchronized_poses, icp_refine=True, manual_refine=True, write_to_file=True):

        frames_dir = os.path.join(scene_dir, "data")

        scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")

        with open(scene_metadata_file, 'r') as file:
            scene_metadata = yaml.safe_load(file)

        camera_name = scene_metadata["camera"]

        camera_intrinsics = load_intrinsics(camera_name)
        camera_distortion = load_distortion(camera_name)
        camera_extrinsics = load_extrinsics(camera_name, scene_dir)

        cam_scale = scene_metadata["cam_scale"]

        sensor_to_virtual_extrinsic = invert_affine(camera_extrinsics)

        #apply extrinsic to convert every pose to actual camera sensor pose
        synchronized_poses_corrected = {}
        for frame_id, synchronized_pose in synchronized_poses.items():
            synchronized_poses_corrected[frame_id] = synchronized_pose @ sensor_to_virtual_extrinsic

        sensor_pose_annotated_frame = synchronized_poses_corrected[annotated_poses_single_frameid]
        sensor_pose_annotated_frame_inv = invert_affine(sensor_pose_annotated_frame)

        object_pcld_transformed = {}
        for obj_id, obj in self._objects.items():
            annotated_obj_pose = annotated_poses_single_frame[obj_id]
            obj_pcld = obj["pcld"].transform(annotated_obj_pose)
            object_pcld_transformed[obj_id] = obj_pcld

        synchronized_poses_refined = {}

        if icp_refine:
            #use first frame coordinate system as world coordinates
            for frame_id, sensor_pose in tqdm(synchronized_poses_corrected.items(), total=len(synchronized_poses_corrected), desc="icp computing refinement"):
                
                rgb = load_rgb(frames_dir, frame_id, "jpg")
                depth = load_depth(frames_dir, frame_id)

                sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ sensor_pose
                sensor_pose_in_annotated_coordinates_inv = invert_affine(sensor_pose_in_annotated_coordinates)

                # First, refine pose
                objects_in_sensor_coords = {}

                for idx, (obj_id, obj_pcld) in enumerate(object_pcld_transformed.items()):
                    obj_pcld = o3d.geometry.PointCloud(obj_pcld) #copy constructor
                    obj_pcld = obj_pcld.transform(sensor_pose_in_annotated_coordinates_inv)
                    objects_in_sensor_coords[obj_id] = obj_pcld
                
                camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsics, camera_distortion)

                #1 -> 2
                pose_refinement_icp = self.refine_pose_icp(list(objects_in_sensor_coords.values()), camera_pcld)

                #2 -> 1, 1 -> opti = 2 -> opti
                synchronized_poses_refined[frame_id] = sensor_pose @ invert_affine(pose_refinement_icp)
        else:
            synchronized_poses_refined = synchronized_poses_corrected

        if manual_refine: 

            #State
            frame_ids = sorted(list(synchronized_poses_refined.keys()))
            frame_ids_idx = 0

            frame_pose = synchronized_poses_refined[frame_ids[frame_ids_idx]]
            sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ frame_pose
            current_pose = invert_affine(sensor_pose_in_annotated_coordinates)

            object_meshes = {}
            for obj_id, obj in self._objects.items():
                annotated_obj_pose = annotated_poses_single_frame[obj_id]
                obj_mesh = obj["mesh"].transform(current_pose @ annotated_obj_pose)
                object_meshes[obj_id] = obj_mesh

            colors = [(80, 225, 116), (74, 118, 56), (194, 193, 120), (176, 216, 249), (214, 251, 255)]
            def render_current_view():

                all_objs_in_sensor_coordinates = []

                for idx, (obj_id, obj_mesh) in enumerate(object_meshes.items()):
                    obj_in_sensor_coordinates = obj_mesh.sample_points_uniformly(number_of_points=10000)
                    all_objs_in_sensor_coordinates.append(obj_in_sensor_coordinates)

                bgr = load_bgr(frames_dir, frame_ids[frame_ids_idx], "jpg")
                
                for idx, obj_pcld_in_sensor_coordinates in enumerate(all_objs_in_sensor_coordinates):

                    obj_pts_in_sensor_coordinates = np.array(obj_pcld_in_sensor_coordinates.points)

                    #(Nx2)
                    obj_pts_projected, _ = cv2.projectPoints(obj_pts_in_sensor_coordinates, np.zeros(3), np.zeros(3), camera_intrinsics, camera_distortion)
                    obj_pts_projected = np.round(obj_pts_projected.squeeze(1)).astype(int)

                    for pt_x, pt_y in obj_pts_projected:
                        bgr = cv2.circle(bgr, (int(pt_x), int(pt_y)), 1, color=colors[idx % len(colors)], thickness=-1)

                return bgr

            cv2.namedWindow("rendered frame")
            cv2.imshow("rendered frame", render_current_view())

            def update_objects():
                nonlocal current_pose
                nonlocal object_meshes

                frame_pose = synchronized_poses_refined[frame_ids[frame_ids_idx]]
                sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ frame_pose
                new_current_pose = invert_affine(sensor_pose_in_annotated_coordinates)

                for obj_id in object_meshes.keys():
                    object_meshes[obj_id] = object_meshes[obj_id].transform(new_current_pose @ invert_affine(current_pose))

                current_pose = new_current_pose

                cv2.imshow("rendered frame", render_current_view())

    #------------------------------------------------------------------------------------------
            def increment_frame_id():
                nonlocal frame_ids_idx
                nonlocal object_meshes

                frame_ids_idx += 1
                frame_ids_idx = frame_ids_idx % len(frame_ids)

                print("frame: {0}".format(frame_ids[frame_ids_idx]))
            
                update_objects()

    #------------------------------------------------------------------------------------------
            def render_side_view():
                all_objs_and_colors_in_sensor_coordinates = []

                for idx, (obj_id, obj_mesh) in enumerate(object_meshes.items()):
                    obj_in_sensor_coordinates = obj_mesh.sample_points_uniformly(number_of_points=10000)
                    all_objs_and_colors_in_sensor_coordinates.append(obj_in_sensor_coordinates, colors[idx % len(colors)])

                bgr = load_bgr(frames_dir, frame_ids[frame_ids_idx], "jpg")
                pcld_in_sensor_coordinates = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsics, camera_distortion, prune_zero=True)

                #side view is a camera off to the right, looking left
                cam2_in_cam1 = np.array([[0, 0, -1, 1], [0, 1, 0, 0], [1, 0, 0, 1.5], [0, 0, 0, 1]])

                pcld_in_cam2_coordinates = pcld_in_sensor_coordinates @ cam2_in_cam1[:3,:3] + cam2_in_cam1[:3,3]

                out = np.zeros_like(bgr)

                scene_pcld_projected, _ = cv2.projectPoints(pcld_in_cam2_coordinates, np.zeros(3), np.zeros(3), camera_intrinsics, camera_distortion)
                scene_pcld_projected = np.round(scene_pcld_projected.squeeze(1)).astype(int)

                for pt_x, pt_y in scene_pcld_projected:
                    out = cv2.circle(out, (int(pt_x), int(pt_y)), 1, color=colors[idx % len(colors)], thickness=-1)

                cv2.imshow("side view", out)
                cv2.waitKey(500)
                cv2.destroyWindow("side view")

    #------------------------------------------------------------------------------------------

            #ROTATION STUFF

            min_rotation_delta = 0.001
            max_rotation_delta = 0.01
            rotation_velocity = 0 #from 0 -> 1
            rotation_velocity_delta = 0.15

            rotation_delta = min_rotation_delta
            last_rot_type = ""

            def rotate_using_euler(euler):
                nonlocal synchronized_poses_refined
                delta_rot_mat = R.from_euler("XYZ", euler).as_matrix()
                synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] = synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] @ delta_rot_mat.T

                update_objects()

            def update_rotation_delta(rot_type):
                nonlocal rotation_velocity
                nonlocal rotation_delta
                nonlocal last_rot_type
                if last_rot_type == rot_type:
                    rotation_velocity = min(rotation_velocity + rotation_velocity_delta, 1)
                else:
                    rotation_velocity = 0
                    last_rot_type = rot_type

                rotation_delta = min_rotation_delta + (max_rotation_delta - min_rotation_delta) * rotation_velocity

            def increase_rotation_alpha():
                update_rotation_delta("incA")
                euler = np.array([rotation_delta, 0, 0])
                rotate_using_euler(euler)

            def decrease_rotation_alpha():
                update_rotation_delta("decA")
                euler = np.array([-rotation_delta, 0, 0])
                rotate_using_euler( euler)

            def increase_rotation_beta():
                update_rotation_delta("incB")
                euler = np.array([0, rotation_delta, 0])
                rotate_using_euler(euler)

            def decrease_rotation_beta():
                update_rotation_delta("decB")
                euler = np.array([0, -rotation_delta, 0])
                rotate_using_euler(euler)

            def increase_rotation_gamma():
                update_rotation_delta("incC")
                euler = np.array([0, 0, rotation_delta])
                rotate_using_euler(euler)
            
            def decrease_rotation_gamma():
                update_rotation_delta("decC")
                euler = np.array([0, 0, -rotation_delta])
                rotate_using_euler(euler)

    #------------------------------------------------------------------------------------------   

            #TRANSLATION STUFF

            min_translation_delta = 0.0005
            max_translation_delta = 0.005
            translation_velocity = 0 #from 0 -> 1
            translation_velocity_delta = 0.05

            translation_delta = min_translation_delta
            last_translation_type = ""

            def translate(trans):
                nonlocal synchronized_poses_refined
                synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,3] += synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] @ trans
                update_objects()

            def update_translation_delta(translation_type):
                nonlocal translation_velocity
                nonlocal translation_delta
                nonlocal last_translation_type
                if last_translation_type == translation_type:
                    translation_velocity = min(translation_velocity + translation_velocity_delta, 1)
                else:
                    translation_velocity = 0
                    last_translation_type = translation_type

                translation_delta = min_translation_delta + (max_translation_delta - min_translation_delta) * translation_velocity

            def increase_x():
                update_translation_delta("incX")
                trans = np.array([translation_delta, 0, 0])
                translate(trans)

            def decrease_x():
                update_translation_delta("decX")
                trans = np.array([-translation_delta, 0, 0])
                translate(trans)

            def increase_y():
                update_translation_delta("incY")
                trans = np.array([0, translation_delta, 0])
                translate(trans)

            def decrease_y():
                update_translation_delta("decY")
                trans = np.array([0, -translation_delta, 0])
                translate(trans)

            def increase_z():
                update_translation_delta("incZ")
                trans = np.array([0, 0, translation_delta])
                translate(trans)

            def decrease_z():
                update_translation_delta("decZ")
                trans = np.array([0, 0, -translation_delta])
                translate(trans)

    #------------------------------------------------------------------------------------------   
            #MAIN LOOP
            while True:
                k = cv2.waitKey(0)
                if k == ord('b'):
                    break
                elif k == 13: #enter
                    increment_frame_id()
                elif k == ord('r'):
                    synchronized_poses_refined[frame_ids[frame_ids_idx]] = synchronized_poses_corrected[frame_ids[frame_ids_idx]]
                elif k == ord('u'):
                    increase_rotation_alpha()
                elif k == ord('i'):
                    decrease_rotation_alpha()
                elif k == ord('o'):
                    increase_rotation_beta()
                elif k == ord('p'):
                    decrease_rotation_beta()
                elif k == ord('k'):
                    increase_rotation_gamma()
                elif k == ord('l'):
                    decrease_rotation_gamma()
                elif k == ord('e'):
                    decrease_y()
                elif k == ord('q'):
                    increase_y()
                elif k == ord('w'):
                    increase_z()
                elif k == ord('s'):
                    decrease_z()
                elif k == ord('d'):
                    decrease_x()
                elif k == ord('a'):
                    increase_x()

        # change of coordinates for synchronized poses from:
        # (sensor -> opti) back to (virtual -> opti)
        synchronized_poses_refined = {frame_id: sensor_pose @ camera_extrinsics for frame_id, sensor_pose in synchronized_poses_refined.items()}
        synchronized_poses_refined_df = convert_pose_dict_to_df(synchronized_poses_refined)

        if write_to_file:
            self.save_refined_poses_df(scene_dir, synchronized_poses_refined_df)

        return synchronized_poses_refined
