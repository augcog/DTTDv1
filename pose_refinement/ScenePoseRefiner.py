"""
Refine camera pose using manual pose annotation.
"""

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
from utils.frame_utils import load_depth, load_rgb
from utils.mesh_utils import uniformly_sample_mesh_with_textures_as_colors
from utils.pointcloud_utils import pointcloud_from_rgb_depth
from utils.pose_dataframe_utils import convert_pose_dict_to_df

class ScenePoseRefiner():
    def __init__(self, objects, number_of_points=10000):
        self._objects = {}
        for obj_id, obj_data in objects.items():
            obj_pcld = uniformly_sample_mesh_with_textures_as_colors(obj_data["mesh"], obj_data["texture"], number_of_points)
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
        criteria.max_iteration = 4

        pose_correction = o3d.registration.registration_icp(
            all_object_pcld, camera_pcld, .01, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            criteria)

        pose_correction = pose_correction.transformation
        return pose_correction

    @staticmethod
    def save_refined_poses_df(scene_dir, df):
        refined_path = os.path.join(scene_dir, "camera_poses", "camera_poses_synchronized_refined.csv")
        df.to_csv(refined_path)

    def refine_poses(self, scene_dir, annotated_poses_single_frameid, annotated_poses_single_frame, synchronized_poses, icp_refine=True, manual_refine=False, write_to_file=True):

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
                
                rgb = load_rgb(frames_dir, frame_id)
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
            # Setting up visualizer for manual refinement      
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            vis.get_render_option().background_color = np.asarray([0, 0, 0])
            render_option = vis.get_render_option()
            render_option.point_color_option = o3d.visualization.PointColorOption.Color

            view_control = vis.get_view_control()

            #State
            frame_ids = sorted(list(synchronized_poses_refined.keys()))
            frame_ids_idx = 0
            objects_visible = True

            frame_pose = synchronized_poses_refined[frame_ids[frame_ids_idx]]
            sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ frame_pose
            current_pose = invert_affine(sensor_pose_in_annotated_coordinates)

            object_meshes = {}
            for obj_id, obj in self._objects.items():
                annotated_obj_pose = annotated_poses_single_frame[obj_id]
                obj_mesh = obj["mesh"].transform(current_pose @ annotated_obj_pose)
                object_meshes[obj_id] = obj_mesh

            rgb = load_rgb(frames_dir, frame_ids[frame_ids_idx])
            depth = load_depth(frames_dir, frame_ids[frame_ids_idx])

            camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsics, camera_distortion)

            #add camera pointcloud
            vis.add_geometry(camera_pcld)

            for obj in object_meshes.values():
                vis.add_geometry(obj)

            def update_objects(vis):
                nonlocal current_pose
                nonlocal object_meshes

                frame_pose = synchronized_poses_refined[frame_ids[frame_ids_idx]]
                sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ frame_pose
                new_current_pose = invert_affine(sensor_pose_in_annotated_coordinates)

                print("old pose", current_pose)
                print("new pose", new_current_pose)

                for obj_id in object_meshes.keys():
                    object_meshes[obj_id] = object_meshes[obj_id].transform(new_current_pose @ invert_affine(current_pose))
                    vis.update_geometry(object_meshes[obj_id])

                current_pose = new_current_pose

            #SETUP KEY CALLBACKS
    #------------------------------------------------------------------------------------------
            #PRESS 1 to decrement frameid
            def decrement_frame_id(vis):
                nonlocal frame_ids_idx
                nonlocal camera_pcld
                nonlocal object_meshes
                
                vis.remove_geometry(camera_pcld, reset_bounding_box=False)

                frame_ids_idx -= 1
                frame_ids_idx = frame_ids_idx + len(frame_ids) % len(frame_ids)

                print("frame: {0}".format(frame_ids[frame_ids_idx]))
                
                rgb = load_rgb(frames_dir, frame_ids[frame_ids_idx])
                depth = load_depth(frames_dir, frame_ids[frame_ids_idx])

                camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsics, camera_distortion)

                vis.add_geometry(camera_pcld, reset_bounding_box=False)

                update_objects(vis)

                return True

            vis.register_key_callback(ord("1"), partial(decrement_frame_id))

    #------------------------------------------------------------------------------------------
            #PRESS 2 to increment frameid
            def increment_frame_id(vis):
                nonlocal frame_ids_idx
                nonlocal camera_pcld
                nonlocal object_meshes
                
                vis.remove_geometry(camera_pcld, reset_bounding_box=False)

                frame_ids_idx += 1
                frame_ids_idx = frame_ids_idx % len(frame_ids)

                print("frame: {0}".format(frame_ids[frame_ids_idx]))
                
                rgb = load_rgb(frames_dir, frame_ids[frame_ids_idx])
                depth = load_depth(frames_dir, frame_ids[frame_ids_idx])

                camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsics, camera_distortion)

                vis.add_geometry(camera_pcld, reset_bounding_box=False)

                update_objects(vis)

                return True

            vis.register_key_callback(ord("2"), partial(increment_frame_id))

    #------------------------------------------------------------------------------------------
            #PRESS 3 to toggle object visibilities
            def toggle_object_visibilities(vis):

                print("toggling object vis!")

                nonlocal objects_visible
                if objects_visible:
                    for obj_id, obj_mesh in object_meshes.items():
                        vis.remove_geometry(obj_mesh, reset_bounding_box=False)
                else:
                    for obj_id, obj_mesh in object_meshes.items():
                        vis.add_geometry(obj_mesh, reset_bounding_box=False)
                objects_visible = not objects_visible
                return True

            vis.register_key_callback(ord("3"), partial(toggle_object_visibilities))

    #------------------------------------------------------------------------------------------

            #ROTATION STUFF

            min_rotation_delta = 0.001
            max_rotation_delta = 0.01
            rotation_velocity = 0 #from 0 -> 1
            rotation_velocity_delta = 0.15

            rotation_delta = min_rotation_delta
            last_rot_type = ""

            def rotate_using_euler(vis, euler):
                nonlocal synchronized_poses_refined
                delta_rot_mat = R.from_euler("XYZ", euler).as_matrix()
                synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] = synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] @ delta_rot_mat.T

                update_objects(vis)

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

            #PRESS U to increase alpha (euler angle rotation)
            def increase_rotation_alpha(vis):
                update_rotation_delta("incA")
                euler = np.array([rotation_delta, 0, 0])
                rotate_using_euler(vis, euler)
                return True
            
            vis.register_key_callback(ord("U"), partial(increase_rotation_alpha))

            #PRESS I to decrease alpha (euler angle rotation)
            def decrease_rotation_alpha(vis):
                update_rotation_delta("decA")
                euler = np.array([-rotation_delta, 0, 0])
                rotate_using_euler(vis, euler)
                return True
            
            vis.register_key_callback(ord("I"), partial(decrease_rotation_alpha))

            #PRESS O to increase beta (euler angle rotation)
            def increase_rotation_beta(vis):
                update_rotation_delta("incB")
                euler = np.array([0, rotation_delta, 0])
                rotate_using_euler(vis, euler)
                return True
            
            vis.register_key_callback(ord("O"), partial(increase_rotation_beta))

            #PRESS P to decrease beta (euler angle rotation)
            def decrease_rotation_beta(vis):
                update_rotation_delta("decB")
                euler = np.array([0, -rotation_delta, 0])
                rotate_using_euler(vis, euler)
                return True
            
            vis.register_key_callback(ord("P"), partial(decrease_rotation_beta))

            #PRESS K to increase gamma (euler angle rotation)
            def increase_rotation_gamma(vis):
                update_rotation_delta("incC")
                euler = np.array([0, 0, rotation_delta])
                rotate_using_euler(vis, euler)
                return True
            
            vis.register_key_callback(ord("K"), partial(increase_rotation_gamma))

            #PRESS L to decrease beta (euler angle rotation)
            def decrease_rotation_gamma(vis):
                update_rotation_delta("decC")
                euler = np.array([0, 0, -rotation_delta])
                rotate_using_euler(vis, euler)
                return True
            
            vis.register_key_callback(ord("L"), partial(decrease_rotation_gamma))

    #------------------------------------------------------------------------------------------   

            #TRANSLATION STUFF

            min_translation_delta = 0.0005
            max_translation_delta = 0.005
            translation_velocity = 0 #from 0 -> 1
            translation_velocity_delta = 0.05

            translation_delta = min_translation_delta
            last_translation_type = ""

            def translate(vis, trans):
                nonlocal synchronized_poses_refined
                synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,3] += synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] @ trans
                update_objects(vis)

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

            #PRESS H to increase X
            def increase_x(vis):
                update_translation_delta("incX")
                trans = np.array([translation_delta, 0, 0])
                translate(vis, trans)
                return True
            
            vis.register_key_callback(ord("H"), partial(increase_x))

            #PRESS F to decrease X
            def decrease_x(vis):
                update_translation_delta("decX")
                trans = np.array([-translation_delta, 0, 0])
                translate(vis, trans)
                return True
            
            vis.register_key_callback(ord("F"), partial(decrease_x))

            #PRESS B to increase Y
            def increase_y(vis):
                update_translation_delta("incY")
                trans = np.array([0, translation_delta, 0])
                translate(vis, trans)
                return True
            
            vis.register_key_callback(ord("B"), partial(increase_y))

            #PRESS G to decrease Y
            def decrease_y(vis):
                update_translation_delta("decY")
                trans = np.array([0, -translation_delta, 0])
                translate(vis, trans)
                return True
            
            vis.register_key_callback(ord("G"), partial(decrease_y))

            #PRESS N to increase Z
            def increase_z(vis):
                update_translation_delta("incZ")
                trans = np.array([0, 0, translation_delta])
                translate(vis, trans)
                return True
            
            vis.register_key_callback(ord("N"), partial(increase_z))

            #PRESS V to decrease Z
            def decrease_z(vis):
                update_translation_delta("decZ")
                trans = np.array([0, 0, -translation_delta])
                translate(vis, trans)
                return True
            
            vis.register_key_callback(ord("V"), partial(decrease_z))

    #------------------------------------------------------------------------------------------   

            vis.run()
            vis.destroy_window()

        # change of coordinates for synchronized poses from:
        # (sensor -> opti) back to (virtual -> opti)
        synchronized_poses_refined = {frame_id: sensor_pose @ camera_extrinsics for frame_id, sensor_pose in synchronized_poses_refined.items()}
        synchronized_poses_refined_df = convert_pose_dict_to_df(synchronized_poses_refined)

        if write_to_file:
            self.save_refined_poses_df(scene_dir, synchronized_poses_refined_df)

        return synchronized_poses_refined
