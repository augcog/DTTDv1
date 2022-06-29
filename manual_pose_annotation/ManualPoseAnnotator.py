"""
Goal: Write a manual pose annotator for the dataset.

Can initialize the pose using 
1) algorithm
2) ICP
3) previous annotation
4) etc.

"""

import cv2
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import yaml

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from pose_refinement import ScenePoseRefiner
from utils.affine_utils import invert_affine
from utils.camera_utils import load_distortion, load_extrinsics, load_intrinsics, write_archive_extrinsic 
from utils.depth_utils import fill_missing
from utils.frame_utils import load_bgr, load_rgb, load_depth, load_o3d_rgb, load_o3d_depth
from utils.pointcloud_utils import pointcloud_from_rgb_depth

class ManualPoseAnnotator:

    """
    objects contains triangle meshes for each object
    object coordinates are in meters
    """
    def __init__(self, objects):
        self._objects = {}
        for obj_id, obj_data in objects.items():
            self._objects[obj_id] = {"name" : obj_data["name"]}
            self._objects[obj_id]["mesh"] = o3d.geometry.TriangleMesh(obj_data["mesh"]) #copy the geometry

    @staticmethod
    def get_pcld_click_xyz(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients, objects):

        print('Please select the approximate center of each object in the scene')
        
        depth_smoothed = fill_missing(depth, 1 / cam_scale, 1)

        smoothed_camera_pcld = pointcloud_from_rgb_depth(rgb, depth_smoothed, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients, prune_zero=False)

        object_centers = {}

        for obj_id, obj_data in objects.items():
            print("Select the center for object named {0}".format(obj_data["name"]))

            plt.imshow(rgb)
            x, y = plt.ginput(1)[0]
            plt.close()

            print("selected point", int(x), int(y))

            object_centers[obj_id] = np.array(smoothed_camera_pcld.points)[int(int(y) * depth.shape[1] + int(x))]

        return object_centers


    """
    icp pose initializer includes several steps:
    1) Prompt user to click on the RGB image wherever the center of the objects are
    2) Use Open3D's colored ICP to do some more fine-tuning
    """
    @staticmethod
    def icp_pose_initializer(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients, objects):
        
        camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients)

        object_centers = ManualPoseAnnotator.get_pcld_click_xyz(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients, objects)

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
                target_pcld, camera_pcld, .02, trans_init,
                o3d.registration.TransformationEstimationPointToPoint())

            transform_obj_to_scene = transform_obj_to_scene.transformation

            print("icp result for obj {0}".format(obj_id))
            print(transform_obj_to_scene)

            object_pose_initializations[obj_id] = transform_obj_to_scene

        return object_pose_initializations

    @staticmethod
    def previous_initializer(scene_dir, *args):
        previous_annotation_file = os.path.join(scene_dir, "annotated_object_poses", "annotated_object_poses.yaml")
        with open(previous_annotation_file, 'r') as f:
            annotated_poses = yaml.safe_load(f)

        annotated_poses = annotated_poses["object_poses"]

        annotated_poses = {k: np.array(v) for k, v in annotated_poses.items()}

        return annotated_poses

    """
    Skip the ICP. Just put the object where the user clicks.
    """
    @staticmethod
    def point_pose_initializer(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients, objects):
        
        object_centers = ManualPoseAnnotator.get_pcld_click_xyz(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients, objects)

        object_pose_initializations = {}

        for obj_id in object_centers.keys():
            obj_center = object_centers[obj_id]

            trans_init = np.eye(4)
            trans_init[:3,3] = obj_center

            object_pose_initializations[obj_id] = trans_init

        return object_pose_initializations

    """
    rgb and depth are images
    cam_scale is the scale of the depth image, we multiply depth * cam_scale to get depth in meters
    initialization method is a function that takes in rgb, depth, depth scale, camera intrinsic matrix, 
    camera distortion coefficients, and list of object triangle meshes
    and returns a dict {obj_id: transform initialization}, the affine transform for each object
    """
    def annotate_pose(self, scene_dir, synchronized_poses, frameid, initialization_method=None, use_archive_extrinsic=True):

        frames_dir = os.path.join(scene_dir, "data")

        scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")

        with open(scene_metadata_file, 'r') as file:
            scene_metadata = yaml.safe_load(file)

        camera_name = scene_metadata["camera"]
        cam_scale = scene_metadata["cam_scale"]
        num_frames = scene_metadata["num_frames"]

        camera_intrinsic_matrix = load_intrinsics(camera_name)
        camera_distortion_coefficients = load_distortion(camera_name)
        camera_extrinsic = load_extrinsics(camera_name, scene_dir, use_archive=use_archive_extrinsic)

        rgb = load_rgb(frames_dir, frameid)
        depth = load_depth(frames_dir, frameid)
        h, w, _ = rgb.shape

        camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients)

        if initialization_method:
            initial_poses = initialization_method(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients, self._objects)
        else:
            initial_poses = {}
            for obj_id in self._objects.keys():
                initial_poses[obj_id] = np.eye(4)

        print("initialized poses", initial_poses)

        #compute 3d reconstruction of scene in first_frame_id coordinates

        sensor_to_virtual = invert_affine(camera_extrinsic)

        corrected_synchronized_poses = {}
        for frame_id, virtual_to_opti in synchronized_poses.items():
            corrected_synchronized_poses[frame_id] = virtual_to_opti @ sensor_to_virtual

        first_frame_id_pose_inv = invert_affine(corrected_synchronized_poses[frameid]) #opti -> first frame id pose

        poses_in_first_frame_id_coords = {}
        for frame_id, sensor_to_opti in corrected_synchronized_poses.items():
            poses_in_first_frame_id_coords[frame_id] = first_frame_id_pose_inv @ sensor_to_opti

        #settings for 3d recon
        max_depth = 5.00
        voxel_size = 0.007
        sdf_trunc = voxel_size * 5

        intr = o3d.camera.PinholeCameraIntrinsic()
        intr.set_intrinsics(w, h, camera_intrinsic_matrix[0][0], camera_intrinsic_matrix[1][1], camera_intrinsic_matrix[0][2], camera_intrinsic_matrix[1][2])

        volume = o3d.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8)

        for frame_id in range(50):
            rgb = load_o3d_rgb(frames_dir, frame_id)
            depth = load_o3d_depth(frames_dir, frame_id)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, depth, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            extr = invert_affine(poses_in_first_frame_id_coords[frame_id])
            volume.integrate(rgbd_image, intr, extr)

        camera_recon = volume.extract_triangle_mesh()

        camera_representation_switch = 0
        camera_representations = [camera_pcld, camera_recon]
            
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        render_option = vis.get_render_option()
        render_option.point_color_option = o3d.visualization.PointColorOption.Color

        view_control = vis.get_view_control()

        #add camera pointcloud
        vis.add_geometry(camera_pcld)

        #State
        annotated_poses = initial_poses
        object_ids = list(annotated_poses.keys())

        active_obj_idx = 0
        object_meshes = {}
        objects_visible = True
        show_all_objects = True

        curr_frameid = frameid
        frame_skip = 20
        
        for obj_id, obj_data in self._objects.items():
            obj_mesh = obj_data["mesh"]
            obj_mesh = obj_mesh.transform(annotated_poses[obj_id])
            object_meshes[obj_id] = obj_mesh
            
            vis.add_geometry(obj_mesh)


        #SETUP KEY CALLBACKS
#------------------------------------------------------------------------------------------
        def increment_active_obj_idx(vis):

            nonlocal active_obj_idx

            if not show_all_objects:
                vis.remove_geometry(object_meshes[object_ids[active_obj_idx]], reset_bounding_box=False)

            active_obj_idx += 1
            active_obj_idx = active_obj_idx % len(object_ids)

            if not show_all_objects:
                vis.add_geometry(object_meshes[object_ids[active_obj_idx]], reset_bounding_box=False)

            print("switched to modifying object {0}".format(self._objects[object_ids[active_obj_idx]]["name"]))

            return not show_all_objects

        vis.register_key_callback(ord("1"), partial(increment_active_obj_idx))

#------------------------------------------------------------------------------------------
        def toggle_object_visibilities(vis):

            nonlocal objects_visible
            if objects_visible:
                if show_all_objects:
                    for obj_id, obj_mesh in object_meshes.items():
                        vis.remove_geometry(obj_mesh, reset_bounding_box=False)
                else:
                    vis.remove_geometry(object_meshes[object_ids[active_obj_idx]], reset_bounding_box=False)
            else:
                if show_all_objects:
                    for obj_id, obj_mesh in object_meshes.items():
                        vis.add_geometry(obj_mesh, reset_bounding_box=False)
                else:
                    vis.add_geometry(object_meshes[object_ids[active_obj_idx]], reset_bounding_box=False)

            objects_visible = not objects_visible
            return True

        vis.register_key_callback(ord("2"), partial(toggle_object_visibilities))

#------------------------------------------------------------------------------------------
        def toggle_show_all_objects(vis):

            nonlocal show_all_objects

            if objects_visible:
                if show_all_objects:
                    for obj_id, obj_mesh in object_meshes.items():
                        if obj_id == object_ids[active_obj_idx]:
                            continue
                        vis.remove_geometry(obj_mesh, reset_bounding_box=False)
                else:
                    for obj_id, obj_mesh in object_meshes.items():
                        if obj_id == object_ids[active_obj_idx]:
                            continue
                        vis.add_geometry(obj_mesh, reset_bounding_box=False)

            show_all_objects = not show_all_objects
            return objects_visible

        vis.register_key_callback(ord("3"), partial(toggle_show_all_objects))

#------------------------------------------------------------------------------------------
        def toggle_scene_representation(vis):

            nonlocal camera_representation_switch

            vis.remove_geometry(camera_representations[camera_representation_switch], reset_bounding_box=False)
            camera_representation_switch = 1 - camera_representation_switch
            vis.add_geometry(camera_representations[camera_representation_switch], reset_bounding_box=False)
            return True

        vis.register_key_callback(ord("4"), partial(toggle_scene_representation))

#------------------------------------------------------------------------------------------
        def increase_frameid(vis):

            nonlocal curr_frameid
            nonlocal annotated_poses
            nonlocal camera_representations

            new_frameid = (curr_frameid + frame_skip) % num_frames
            rgb = load_rgb(frames_dir, new_frameid)
            depth = load_depth(frames_dir, new_frameid)

            camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients)

            vis.remove_geometry(camera_representations[0], reset_bounding_box=False)
            camera_representations[0] = camera_pcld
            vis.add_geometry(camera_representations[0], reset_bounding_box=False)

            old_cam_pose = corrected_synchronized_poses[curr_frameid]
            new_cam_pose = corrected_synchronized_poses[new_frameid]

            old_to_new = invert_affine(new_cam_pose) @ old_cam_pose

            for obj_id in object_meshes.keys():
                object_meshes[obj_id] = object_meshes[obj_id].transform(old_to_new)
                vis.update_geometry(object_meshes[obj_id])

            for obj_id in annotated_poses.keys():
                annotated_poses[obj_id] = old_to_new @ annotated_poses[obj_id]

            curr_frameid = new_frameid

            return True

        vis.register_key_callback(ord("6"), partial(increase_frameid))

#------------------------------------------------------------------------------------------
        def decrease_frameid(vis):

            nonlocal curr_frameid
            nonlocal annotated_poses
            nonlocal camera_representations

            new_frameid = (curr_frameid - frame_skip + num_frames) % num_frames
            rgb = load_rgb(frames_dir, new_frameid)
            depth = load_depth(frames_dir, new_frameid)

            camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients)

            vis.remove_geometry(camera_representations[0], reset_bounding_box=False)
            camera_representations[0] = camera_pcld
            vis.add_geometry(camera_representations[0], reset_bounding_box=False)

            old_cam_pose = corrected_synchronized_poses[curr_frameid]
            new_cam_pose = corrected_synchronized_poses[new_frameid]

            old_to_new = invert_affine(new_cam_pose) @ old_cam_pose

            for obj_id in object_meshes.keys():
                object_meshes[obj_id] = object_meshes[obj_id].transform(old_to_new)
                vis.update_geometry(object_meshes[obj_id])

            for obj_id in annotated_poses.keys():
                annotated_poses[obj_id] = old_to_new @ annotated_poses[obj_id]

            curr_frameid = new_frameid

            return True

        vis.register_key_callback(ord("5"), partial(decrease_frameid))

#------------------------------------------------------------------------------------------
        def return_to_start_frame(vis):

            nonlocal curr_frameid
            nonlocal annotated_poses
            nonlocal camera_representations

            new_frameid = frameid
            rgb = load_rgb(frames_dir, new_frameid)
            depth = load_depth(frames_dir, new_frameid)

            camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsic_matrix, camera_distortion_coefficients)

            vis.remove_geometry(camera_representations[0], reset_bounding_box=False)
            camera_representations[0] = camera_pcld
            vis.add_geometry(camera_representations[0], reset_bounding_box=False)

            old_cam_pose = corrected_synchronized_poses[curr_frameid]
            new_cam_pose = corrected_synchronized_poses[new_frameid]

            old_to_new = invert_affine(new_cam_pose) @ old_cam_pose

            for obj_id in object_meshes.keys():
                object_meshes[obj_id] = object_meshes[obj_id].transform(old_to_new)
                vis.update_geometry(object_meshes[obj_id])

            for obj_id in annotated_poses.keys():
                annotated_poses[obj_id] = old_to_new @ annotated_poses[obj_id]

            curr_frameid = new_frameid

            return True

        vis.register_key_callback(ord("7"), partial(return_to_start_frame))

#------------------------------------------------------------------------------------------
        def icp_current_obj(vis):

            nonlocal annotated_poses

            criteria = o3d.registration.ICPConvergenceCriteria()
            criteria.max_iteration = 2

            obj_pcld = object_meshes[object_ids[active_obj_idx]].sample_points_uniformly(number_of_points=1000)

            pose_correction = o3d.registration.registration_icp(
                obj_pcld, camera_representations[0], .01, np.eye(4),
                o3d.registration.TransformationEstimationPointToPoint(),
                criteria)

            pose_correction = pose_correction.transformation

            object_meshes[object_ids[active_obj_idx]] = object_meshes[object_ids[active_obj_idx]].transform(pose_correction)
            new_annotated_pose = np.copy(annotated_poses[object_ids[active_obj_idx]])
            annotated_poses[object_ids[active_obj_idx]] = pose_correction @ new_annotated_pose
            vis.update_geometry(object_meshes[object_ids[active_obj_idx]])

            return True

        vis.register_key_callback(ord(" "), partial(icp_current_obj))

#------------------------------------------------------------------------------------------

        colors = [(80, 225, 116), (74, 118, 56), (194, 193, 120), (176, 216, 249), (214, 251, 255)]

        def render_current_view(refine, vis):

            all_objs_in_sensor_coordinates = []

            for idx, (obj_id, obj_mesh) in enumerate(object_meshes.items()):
                obj_in_sensor_coordinates = obj_mesh.sample_points_uniformly(number_of_points=10000)
                all_objs_in_sensor_coordinates.append(obj_in_sensor_coordinates)

            if refine:
                #perform camera pose refinement to simulate final result

                scene_pose_refiner = ScenePoseRefiner()
                refinement = scene_pose_refiner.refine_pose_icp(all_objs_in_sensor_coordinates, camera_representations[0])    

                print("camera refinement: ", refinement)

                for idx in range(len(all_objs_in_sensor_coordinates)):
                    all_objs_in_sensor_coordinates[idx] = all_objs_in_sensor_coordinates[idx].transform(refinement)

            bgr = load_bgr(frames_dir, curr_frameid)
            
            for idx, obj_pcld_in_sensor_coordinates in enumerate(all_objs_in_sensor_coordinates):

                obj_pts_in_sensor_coordinates = np.array(obj_pcld_in_sensor_coordinates.points)

                #(Nx2)
                obj_pts_projected, _ = cv2.projectPoints(obj_pts_in_sensor_coordinates, np.zeros(3), np.zeros(3), camera_intrinsic_matrix, camera_distortion_coefficients)
                obj_pts_projected = np.round(obj_pts_projected.squeeze(1)).astype(int)

                for pt_x, pt_y in obj_pts_projected:
                    bgr = cv2.circle(bgr, (int(pt_x), int(pt_y)), 1, color=colors[idx % len(colors)], thickness=-1)

            cv2.imshow("rendered view", bgr)
            cv2.waitKey(1000)
            cv2.destroyWindow("rendered view")

            return False


        vis.register_key_callback(ord("Z"), partial(render_current_view, False))

        vis.register_key_callback(ord("X"), partial(render_current_view, True))

#------------------------------------------------------------------------------------------

        def align_to_ground_plane(vis):

            camera_pcld = camera_representations[0]

            obj_id = object_ids[active_obj_idx]
            obj_pcld = object_meshes[obj_id].sample_points_uniformly(number_of_points=10000)

            obj_pts = np.copy(np.array(obj_pcld.points))

            while True:
                plane_model, inliers = camera_pcld.segment_plane(distance_threshold=0.007,
                                                    ransac_n=3,
                                                    num_iterations=1000)

                [a, b, c, d] = plane_model

                #ensure plane normal is pointing up (assume gravity is downwards (positive Y))
                if np.array([a, b, c]) @ np.array([0, -1, 0]) < 0:

                    print("flipping normal")

                    a = -a
                    b = -b
                    c = -c
                    d = -d

                plane_normal = np.array([a, b, c])

                closest_pt = np.min(np.abs(a * obj_pts[:,0] + b * obj_pts[:,1] + c * obj_pts[:,2] + d))

                #might detect a plane outside of the table
                if closest_pt > 0.05:
                    camera_pcld = camera_pcld.select_by_index(inliers, inverse=True)
                else:
                    break

            
            most_underground = np.min(a * obj_pts[:,0] + b * obj_pts[:,1] + c * obj_pts[:,2] + d)
            upwards_delta = np.array([a, b, c]) * min(most_underground, 0) * -1

            print("moving upwards", upwards_delta)

            obj_pts += upwards_delta

            obj_pts_touching_ground_mask = np.abs(a * obj_pts[:,0] + b * obj_pts[:,1] + c * obj_pts[:,2] + d) < 0.001

            obj_pts_touching_ground = obj_pts[obj_pts_touching_ground_mask]

            num_pts = obj_pts_touching_ground.shape[0]

            dists = cdist(obj_pts_touching_ground, obj_pts_touching_ground)
            dists = dists.flatten()

            furthest_pts = np.argmax(dists)

            furthest_pt_1 = int(furthest_pts / num_pts)
            furthest_pt_2 = int(furthest_pts % num_pts)

            dists = dists.reshape((num_pts, num_pts))

            pt_1 = obj_pts_touching_ground[furthest_pt_1]
            pt_2 = obj_pts_touching_ground[furthest_pt_2]

            furthest_vec = pt_2 - pt_1
            furthest_vec /= np.linalg.norm(furthest_vec, keepdims=True)
            
            projection = furthest_vec - furthest_vec @ plane_normal / np.square(np.linalg.norm(plane_normal)) * plane_normal

            projection /= np.linalg.norm(projection, keepdims=True)

            rotation = np.arccos(furthest_vec @ projection)

            print("rotating by radians", rotation)

            rot_vec = np.cross(furthest_vec, projection)
            rot_vec /= np.linalg.norm(projection, keepdims=True)

            rot_mat = R.from_rotvec(rot_vec * rotation).as_matrix()

            affine = np.eye(4)
            affine[:3,:3] = rot_mat
            affine[:3,3] = upwards_delta

            object_meshes[obj_id] = object_meshes[obj_id].transform(affine)
            annotated_poses[obj_id] = affine @ annotated_poses[obj_id]

            vis.update_geometry(object_meshes[obj_id])

            return True

        vis.register_key_callback(ord("C"), partial(align_to_ground_plane))
#------------------------------------------------------------------------------------------

        #ROTATION STUFF

        min_rotation_delta = 0.002
        max_rotation_delta = 0.07
        rotation_velocity = 0 #from 0 -> 1
        rotation_velocity_delta = 0.15

        rotation_delta = min_rotation_delta
        last_rot_type = ""

        def rotate_using_euler(vis, euler):
            nonlocal annotated_poses
            nonlocal active_obj_idx

            delta_rot_mat = R.from_euler("XYZ", euler).as_matrix()
            current_rot_mat = annotated_poses[object_ids[active_obj_idx]][:3,:3]
            object_meshes[object_ids[active_obj_idx]] = object_meshes[object_ids[active_obj_idx]].rotate(current_rot_mat @ delta_rot_mat @ current_rot_mat.T, annotated_poses[object_ids[active_obj_idx]][:3,3])
            new_annotated_pose = np.copy(annotated_poses[object_ids[active_obj_idx]])
            new_annotated_pose[:3,:3] = current_rot_mat @ delta_rot_mat
            annotated_poses[object_ids[active_obj_idx]] = new_annotated_pose
            vis.update_geometry(object_meshes[object_ids[active_obj_idx]])

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

        def increase_rotation_alpha(vis):
            update_rotation_delta("incA")
            euler = np.array([rotation_delta, 0, 0])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("U"), partial(increase_rotation_alpha))

        def decrease_rotation_alpha(vis):
            update_rotation_delta("decA")
            euler = np.array([-rotation_delta, 0, 0])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("I"), partial(decrease_rotation_alpha))

        def increase_rotation_beta(vis):
            update_rotation_delta("incB")
            euler = np.array([0, rotation_delta, 0])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("O"), partial(increase_rotation_beta))

        def decrease_rotation_beta(vis):
            update_rotation_delta("decB")
            euler = np.array([0, -rotation_delta, 0])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("P"), partial(decrease_rotation_beta))

        def increase_rotation_gamma(vis):
            update_rotation_delta("incC")
            euler = np.array([0, 0, rotation_delta])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("K"), partial(increase_rotation_gamma))

        def decrease_rotation_gamma(vis):
            update_rotation_delta("decC")
            euler = np.array([0, 0, -rotation_delta])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("L"), partial(decrease_rotation_gamma))

#------------------------------------------------------------------------------------------   

        #TRANSLATION STUFF

        min_translation_delta = 0.0005
        max_translation_delta = 0.01
        translation_velocity = 0 #from 0 -> 1
        translation_velocity_delta = 0.05

        translation_delta = min_translation_delta
        last_translation_type = ""

        def translate(vis, trans):
            nonlocal annotated_poses
            nonlocal active_obj_idx

            current_rot_mat = annotated_poses[object_ids[active_obj_idx]][:3,:3]
            world_trans = current_rot_mat @ trans

            object_meshes[object_ids[active_obj_idx]] = object_meshes[object_ids[active_obj_idx]].translate(world_trans)
            
            new_annotated_pose = np.copy(annotated_poses[object_ids[active_obj_idx]])
            new_annotated_pose[:3,3] += world_trans
            annotated_poses[object_ids[active_obj_idx]] = new_annotated_pose
            vis.update_geometry(object_meshes[object_ids[active_obj_idx]])

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

        def increase_x(vis):
            update_translation_delta("incX")
            trans = np.array([translation_delta, 0, 0])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("D"), partial(increase_x))

        def decrease_x(vis):
            update_translation_delta("decX")
            trans = np.array([-translation_delta, 0, 0])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("A"), partial(decrease_x))

        def increase_y(vis):
            update_translation_delta("incY")
            trans = np.array([0, translation_delta, 0])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("W"), partial(increase_y))

        def decrease_y(vis):
            update_translation_delta("decY")
            trans = np.array([0, -translation_delta, 0])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("S"), partial(decrease_y))

        def increase_z(vis):
            update_translation_delta("incZ")
            trans = np.array([0, 0, translation_delta])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("Q"), partial(increase_z))

        def decrease_z(vis):
            update_translation_delta("decZ")
            trans = np.array([0, 0, -translation_delta])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("E"), partial(decrease_z))

#------------------------------------------------------------------------------------------   

        vis.run()
        vis.destroy_window()

        #transform poses back to annotation frame
        curr_pose = corrected_synchronized_poses[curr_frameid]
        annotated_pose = corrected_synchronized_poses[frameid]

        correction = invert_affine(annotated_pose) @ curr_pose

        for obj_id in annotated_poses.keys():
            annotated_poses[obj_id] = correction @ annotated_poses[obj_id]

        #fix minor collisions

        print("Fixing collisions.")

        annotated_poses_collision = {}
        for obj_id, obj_pose in annotated_poses.items():
            annotated_poses_collision[obj_id] = np.copy(obj_pose)
        
        obj_meshes_collision = {}
        for obj_id, obj_mesh in object_meshes.items():
            obj_mesh = o3d.geometry.TriangleMesh(obj_mesh)
            obj_mesh = obj_mesh.simplify_quadric_decimation(3000)
            obj_meshes_collision[obj_id] = obj_mesh

        number_of_movements = 0

        #each movement pushes objects 1 millimeter from one another
        movement_size = 0.001

        while True:

            moved = False

            for obj_id_1 in annotated_poses_collision.keys():
                for obj_id_2 in annotated_poses_collision.keys():

                    if obj_id_2 <= obj_id_1:
                        continue

                    mesh1 = obj_meshes_collision[obj_id_1]
                    mesh2 = obj_meshes_collision[obj_id_2]
                    if mesh1.is_intersecting(mesh2):
                        mesh1_center = mesh1.get_center()
                        mesh2_center = mesh2.get_center()
                        diff = mesh2_center - mesh1_center

                        print("Objects {0} and {1} are {2} meters away from each other.".format(self._objects[obj_id_1]["name"], self._objects[obj_id_2]["name"], np.linalg.norm(diff)))
                        
                        diff /= np.linalg.norm(diff, keepdims=True)

                        mesh1_translation = -diff * movement_size / 2.
                        mesh2_translation = diff * movement_size / 2.

                        new_pose_1 = np.copy(annotated_poses_collision[obj_id_1])
                        new_pose_1[:3,3] += mesh1_translation
                        annotated_poses_collision[obj_id_1] = new_pose_1

                        new_pose_2 = np.copy(annotated_poses_collision[obj_id_2])
                        new_pose_2[:3,3] += mesh2_translation
                        annotated_poses_collision[obj_id_2] = new_pose_2

                        obj_meshes_collision[obj_id_1] = mesh1.translate(mesh1_translation)
                        obj_meshes_collision[obj_id_2] = mesh2.translate(mesh2_translation)

                        print("Moved {0} and {1} away from each other.".format(self._objects[obj_id_1]["name"], self._objects[obj_id_2]["name"]))

                        number_of_movements += 1
                        moved = True

            if not moved:
                break

        print("Fixed collisions between meshes. Required {0} {1} meter translations.".format(number_of_movements, movement_size))

        if number_of_movements > 5:
            print("WARNING!! This might lead to error, try not to overlap so much!")

        annotated_poses = annotated_poses_collision

        print("archiving extrinsic!")
        write_archive_extrinsic(camera_extrinsic, scene_dir)

        return annotated_poses
