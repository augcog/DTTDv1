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
from functools import partial
from scipy.spatial.transform import Rotation as R

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
    def pointcloud_from_rgb_depth(rgb, depth, depth_scale, intrinsic, distortion, prune_zero=True):

        points = np.array([[(k, i) for k in range(depth.shape[1])] for i in range(depth.shape[0])]).reshape((-1, 2)).astype(np.float32)
        Z = depth.flatten() * depth_scale

        f_x = intrinsic[0, 0]
        f_y = intrinsic[1, 1]
        c_x = intrinsic[0, 2]
        c_y = intrinsic[1, 2]

        rgb = rgb.reshape((-1, 3))

        if prune_zero:
            points = points[Z > 0]
            rgb = rgb[Z > 0]
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
        smoothed_camera_pcld = ManualPoseAnnotator.pointcloud_from_rgb_depth(rgb, depth_smoothed, depth_scale, camera_intrinsic_matrix, camera_distortion_coefficients, prune_zero=False)

        print('Please select the approximate center of each object in the scene')

        object_centers = {}

        for obj_id, obj_data in objects.items():
            print("Select the center for object named {0}".format(obj_data["name"]))

            plt.imshow(rgb)
            x, y = plt.ginput(1)[0]
            plt.close()

            print("selected point", x, y)

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

        print("initialized poses", initial_poses)

        #TODO: Pose Annotator here
        #planning to use open3d.visualization.VisualizerWithKeyCallback to make this happen
 
        camera_pcld = ManualPoseAnnotator.pointcloud_from_rgb_depth(rgb, depth, depth_scale, self.camera_intrinsic_matrix, self.camera_distortion_coefficients)

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

        
        for obj_id, obj_data in self.objects.items():
            obj_mesh = obj_data["mesh"]
            obj_mesh = obj_mesh.transform(annotated_poses[obj_id])
            object_meshes[obj_id] = obj_mesh
            
            vis.add_geometry(obj_mesh)


        #SETUP KEY CALLBACKS
#------------------------------------------------------------------------------------------
        #PRESS 1 to change which object you are annotating
        def increment_active_obj_idx(vis):

            print("incrementing active obj idx!")

            nonlocal active_obj_idx
            active_obj_idx += 1
            active_obj_idx = active_obj_idx % len(object_ids)
            return False

        vis.register_key_callback(ord("1"), partial(increment_active_obj_idx))

#------------------------------------------------------------------------------------------

        #ROTATION STUFF

        min_rotation_delta = 0.005
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
            object_meshes[obj_id] = object_meshes[obj_id].rotate(current_rot_mat @ delta_rot_mat @ current_rot_mat.T, annotated_poses[object_ids[active_obj_idx]][:3,3])
            new_annotated_pose = np.copy(annotated_poses[object_ids[active_obj_idx]])
            new_annotated_pose[:3,:3] = current_rot_mat @ delta_rot_mat
            annotated_poses[object_ids[active_obj_idx]] = new_annotated_pose
            vis.update_geometry(object_meshes[obj_id])

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

        #PRESS I to increase alpha (euler angle rotation)
        def increase_rotation_alpha(vis):
            update_rotation_delta("incA")
            euler = np.array([rotation_delta, 0, 0])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("I"), partial(increase_rotation_alpha))

        #PRESS J to decrease alpha (euler angle rotation)
        def decrease_rotation_alpha(vis):
            update_rotation_delta("decA")
            euler = np.array([-rotation_delta, 0, 0])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("J"), partial(decrease_rotation_alpha))

        #PRESS O to increase beta (euler angle rotation)
        def increase_rotation_beta(vis):
            update_rotation_delta("incB")
            euler = np.array([0, rotation_delta, 0])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("O"), partial(increase_rotation_beta))

        #PRESS K to decrease beta (euler angle rotation)
        def decrease_rotation_beta(vis):
            update_rotation_delta("decB")
            euler = np.array([0, -rotation_delta, 0])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("K"), partial(decrease_rotation_beta))

        #PRESS P to increase gamma (euler angle rotation)
        def increase_rotation_gamma(vis):
            update_rotation_delta("incC")
            euler = np.array([0, 0, rotation_delta])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("P"), partial(increase_rotation_gamma))

        #PRESS L to decrease beta (euler angle rotation)
        def decrease_rotation_gamma(vis):
            update_rotation_delta("decC")
            euler = np.array([0, 0, -rotation_delta])
            rotate_using_euler(vis, euler)
            return True
        
        vis.register_key_callback(ord("L"), partial(decrease_rotation_gamma))

#------------------------------------------------------------------------------------------   

        #TRANSLATION STUFF

        min_translation_delta = 0.005
        max_translation_delta = 0.07
        translation_velocity = 0 #from 0 -> 1
        translation_velocity_delta = 0.15

        translation_delta = min_translation_delta
        last_translation_type = ""

        def translate(vis, trans):
            nonlocal annotated_poses
            nonlocal active_obj_idx

            current_rot_mat = annotated_poses[object_ids[active_obj_idx]][:3,:3]
            world_trans = current_rot_mat @ trans

            object_meshes[obj_id] = object_meshes[obj_id].translate(world_trans)
            
            new_annotated_pose = np.copy(annotated_poses[object_ids[active_obj_idx]])
            new_annotated_pose[:3,3] += world_trans
            annotated_poses[object_ids[active_obj_idx]] = new_annotated_pose
            vis.update_geometry(object_meshes[obj_id])

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

        #PRESS G to increase X
        def increase_x(vis):
            update_translation_delta("incX")
            trans = np.array([translation_delta, 0, 0])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("G"), partial(increase_x))

        #PRESS H to decrease X
        def decrease_x(vis):
            update_translation_delta("decX")
            trans = np.array([-translation_delta, 0, 0])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("H"), partial(decrease_x))

        #PRESS V to increase Y
        def increase_y(vis):
            update_translation_delta("incY")
            trans = np.array([0, translation_delta, 0])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("V"), partial(increase_y))

        #PRESS B to decrease Y
        def decrease_y(vis):
            update_translation_delta("decY")
            trans = np.array([0, -translation_delta, 0])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("B"), partial(decrease_y))

        #PRESS N to increase Z
        def increase_z(vis):
            update_translation_delta("incZ")
            trans = np.array([0, 0, translation_delta])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("N"), partial(increase_z))

        #PRESS M to decrease Z
        def decrease_z(vis):
            update_translation_delta("decZ")
            trans = np.array([0, 0, -translation_delta])
            translate(vis, trans)
            return True
        
        vis.register_key_callback(ord("M"), partial(decrease_z))

#------------------------------------------------------------------------------------------   

        vis.run()
        vis.destroy_window()
        
        return annotated_poses
