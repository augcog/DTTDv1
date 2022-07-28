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
from utils.pointcloud_utils import apply_affine_to_points, pointcloud_from_rgb_depth
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
            synchronized_poses_refined = {k: np.copy(v) for k, v in synchronized_poses_corrected.items()}

        if manual_refine: 

            #State
            frame_ids = [id for id in sorted(list(synchronized_poses_refined.keys())) if id != annotated_poses_single_frameid]
            frame_ids_idx = 0

            frame_pose = synchronized_poses_refined[frame_ids[frame_ids_idx]]
            sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ frame_pose
            current_pose = invert_affine(sensor_pose_in_annotated_coordinates)

            current_refinement = np.eye(4)

            show_objects = True

            object_meshes_and_bbs = {}
            for obj_id, obj in self._objects.items():

                annotated_obj_pose = annotated_poses_single_frame[obj_id]

                obj_bb = obj["mesh"].get_axis_aligned_bounding_box()
                obj_bb_pts = np.array(obj_bb.get_box_points())
                obj_bb_pts = apply_affine_to_points(obj_bb_pts, annotated_obj_pose)

                obj_mesh = obj["mesh"].transform(current_pose @ annotated_obj_pose)
                object_meshes_and_bbs[obj_id] = (obj_mesh, obj_bb_pts)

            colors = [(80, 225, 116), (74, 118, 56), (194, 193, 120), (176, 216, 249), (214, 251, 255)]

            def render_current_view():

                objs_and_bbs_in_sensor_coords = []

                for idx, (obj_id, (obj_mesh, obj_bb)) in enumerate(object_meshes_and_bbs.items()):
                    obj_in_sensor_coordinates = obj_mesh.sample_points_uniformly(number_of_points=10000)
                    objs_and_bbs_in_sensor_coords.append((obj_in_sensor_coordinates, obj_bb))

                bgr = load_bgr(frames_dir, frame_ids[frame_ids_idx], "jpg")

                if show_objects:
                    for idx, (obj_pcld_in_sensor_coordinates, obj_bb) in enumerate(objs_and_bbs_in_sensor_coords):

                        obj_pts_in_sensor_coordinates = np.array(obj_pcld_in_sensor_coordinates.points)

                        #(Nx2)
                        obj_pts_projected, _ = cv2.projectPoints(obj_pts_in_sensor_coordinates, np.zeros(3), np.zeros(3), camera_intrinsics, camera_distortion)
                        obj_pts_projected = np.round(obj_pts_projected.squeeze(1)).astype(int)

                        for pt_x, pt_y in obj_pts_projected:
                            bgr = cv2.circle(bgr, (int(pt_x), int(pt_y)), 1, color=colors[idx % len(colors)], thickness=-1)

                        obj_bb_projected, _ = cv2.projectPoints(obj_bb, np.zeros(3), np.zeros(3), camera_intrinsics, camera_distortion)
                        bb_proj = np.round(obj_bb_projected.squeeze(1)).astype(int)

                        bgr = cv2.line(bgr, (int(bb_proj[0][0]), int(bb_proj[0][1])), (int(bb_proj[1][0]), int(bb_proj[1][1])), color=(0,0,0), thickness=1)
                        bgr = cv2.line(bgr, (int(bb_proj[0][0]), int(bb_proj[0][1])), (int(bb_proj[2][0]), int(bb_proj[2][1])), color=(0,0,0), thickness=1)
                        bgr = cv2.line(bgr, (int(bb_proj[1][0]), int(bb_proj[1][1])), (int(bb_proj[7][0]), int(bb_proj[7][1])), color=(0,0,0), thickness=1)
                        bgr = cv2.line(bgr, (int(bb_proj[2][0]), int(bb_proj[2][1])), (int(bb_proj[7][0]), int(bb_proj[7][1])), color=(0,0,0), thickness=1)

                        bgr = cv2.line(bgr, (int(bb_proj[4][0]), int(bb_proj[4][1])), (int(bb_proj[5][0]), int(bb_proj[5][1])), color=(0,0,0), thickness=1)
                        bgr = cv2.line(bgr, (int(bb_proj[4][0]), int(bb_proj[4][1])), (int(bb_proj[6][0]), int(bb_proj[6][1])), color=(0,0,0), thickness=1)
                        bgr = cv2.line(bgr, (int(bb_proj[3][0]), int(bb_proj[3][1])), (int(bb_proj[5][0]), int(bb_proj[5][1])), color=(0,0,0), thickness=1)
                        bgr = cv2.line(bgr, (int(bb_proj[3][0]), int(bb_proj[3][1])), (int(bb_proj[6][0]), int(bb_proj[6][1])), color=(0,0,0), thickness=1)

                        bgr = cv2.line(bgr, (int(bb_proj[0][0]), int(bb_proj[0][1])), (int(bb_proj[3][0]), int(bb_proj[3][1])), color=(0,0,0), thickness=1)
                        bgr = cv2.line(bgr, (int(bb_proj[1][0]), int(bb_proj[1][1])), (int(bb_proj[6][0]), int(bb_proj[6][1])), color=(0,0,0), thickness=1)
                        bgr = cv2.line(bgr, (int(bb_proj[2][0]), int(bb_proj[2][1])), (int(bb_proj[5][0]), int(bb_proj[5][1])), color=(0,0,0), thickness=1)
                        bgr = cv2.line(bgr, (int(bb_proj[7][0]), int(bb_proj[7][1])), (int(bb_proj[4][0]), int(bb_proj[4][1])), color=(0,0,0), thickness=1)

                return bgr

            alt_view_cache = {}

            def render_alt_view(cam2_in_cam1, view):

                if view in alt_view_cache.keys():
                    return alt_view_cache[view][show_objects]

                objs_bbs_color_in_sensor_coords = []

                for idx, (obj_id, (obj_mesh, obj_bb)) in enumerate(object_meshes_and_bbs.items()):
                    obj_in_sensor_coordinates = np.array(obj_mesh.sample_points_uniformly(number_of_points=10000).points)
                    objs_bbs_color_in_sensor_coords.append((obj_in_sensor_coordinates, obj_bb, colors[idx % len(colors)]))

                bgr = load_bgr(frames_dir, frame_ids[frame_ids_idx], "jpg")
                depth = load_depth(frames_dir, frame_ids[frame_ids_idx])

                pcld_in_sensor_coordinates = pointcloud_from_rgb_depth(bgr, depth, cam_scale, camera_intrinsics, camera_distortion, prune_zero=True)
                points = np.array(pcld_in_sensor_coordinates.points)
                pcld_colors = np.array(pcld_in_sensor_coordinates.colors)

                pcld_in_cam2_coordinates = apply_affine_to_points(points, invert_affine(cam2_in_cam1))

                pcld_in_cam2_coordinates = pcld_in_cam2_coordinates.reshape((-1, 3))
                pcld_colors = (pcld_colors.reshape((-1, 3)) * 255.).astype(int)

                num_pts = pcld_in_cam2_coordinates.shape[0]
                
                sampled_pts = np.random.choice(num_pts, 50000, replace=False)

                pcld_in_cam2_coordinates = pcld_in_cam2_coordinates[sampled_pts]
                pcld_colors = pcld_colors[sampled_pts]

                out = np.zeros_like(bgr)

                scene_pcld_projected, _ = cv2.projectPoints(pcld_in_cam2_coordinates, np.zeros(3), np.zeros(3), camera_intrinsics, camera_distortion)
                scene_pcld_projected = np.round(scene_pcld_projected.squeeze(1)).astype(int)

                for (pt_x, pt_y), (b, g, r) in zip(scene_pcld_projected, pcld_colors):
                    out = cv2.circle(out, (int(pt_x), int(pt_y)), 1, color=(int(b), int(g), int(r)), thickness=-1)

                alt_view_cache[view] = {}
                alt_view_cache[view][False] = np.copy(out)

                for (obj_pts, obj_bb, color) in objs_bbs_color_in_sensor_coords:

                    obj_pts_in_cam2 = apply_affine_to_points(obj_pts, invert_affine(cam2_in_cam1))
                    obj_bb_in_cam2 = apply_affine_to_points(obj_bb, invert_affine(cam2_in_cam1))

                    obj_pts_in_cam2_projected, _ = cv2.projectPoints(obj_pts_in_cam2, np.zeros(3), np.zeros(3), camera_intrinsics, camera_distortion)
                    obj_pts_in_cam2_projected = np.round(obj_pts_in_cam2_projected.squeeze(1)).astype(int)

                    for (pt_x, pt_y) in obj_pts_in_cam2_projected:
                        out = cv2.circle(out, (int(pt_x), int(pt_y)), 1, color=color, thickness=-1)

                    obj_bb_projected, _ = cv2.projectPoints(obj_bb_in_cam2, np.zeros(3), np.zeros(3), camera_intrinsics, camera_distortion)
                    bb_proj = np.round(obj_bb_projected.squeeze(1)).astype(int)

                    out = cv2.line(out, (int(bb_proj[0][0]), int(bb_proj[0][1])), (int(bb_proj[1][0]), int(bb_proj[1][1])), color=(0,255,100), thickness=1)
                    out = cv2.line(out, (int(bb_proj[0][0]), int(bb_proj[0][1])), (int(bb_proj[2][0]), int(bb_proj[2][1])), color=(0,255,100), thickness=1)
                    out = cv2.line(out, (int(bb_proj[1][0]), int(bb_proj[1][1])), (int(bb_proj[7][0]), int(bb_proj[7][1])), color=(0,255,100), thickness=1)
                    out = cv2.line(out, (int(bb_proj[2][0]), int(bb_proj[2][1])), (int(bb_proj[7][0]), int(bb_proj[7][1])), color=(0,255,100), thickness=1)

                    out = cv2.line(out, (int(bb_proj[4][0]), int(bb_proj[4][1])), (int(bb_proj[5][0]), int(bb_proj[5][1])), color=(0,255,100), thickness=1)
                    out = cv2.line(out, (int(bb_proj[4][0]), int(bb_proj[4][1])), (int(bb_proj[6][0]), int(bb_proj[6][1])), color=(0,255,100), thickness=1)
                    out = cv2.line(out, (int(bb_proj[3][0]), int(bb_proj[3][1])), (int(bb_proj[5][0]), int(bb_proj[5][1])), color=(0,255,100), thickness=1)
                    out = cv2.line(out, (int(bb_proj[3][0]), int(bb_proj[3][1])), (int(bb_proj[6][0]), int(bb_proj[6][1])), color=(0,255,100), thickness=1)

                    out = cv2.line(out, (int(bb_proj[0][0]), int(bb_proj[0][1])), (int(bb_proj[3][0]), int(bb_proj[3][1])), color=(0,255,100), thickness=1)
                    out = cv2.line(out, (int(bb_proj[1][0]), int(bb_proj[1][1])), (int(bb_proj[6][0]), int(bb_proj[6][1])), color=(0,255,100), thickness=1)
                    out = cv2.line(out, (int(bb_proj[2][0]), int(bb_proj[2][1])), (int(bb_proj[5][0]), int(bb_proj[5][1])), color=(0,255,100), thickness=1)
                    out = cv2.line(out, (int(bb_proj[7][0]), int(bb_proj[7][1])), (int(bb_proj[4][0]), int(bb_proj[4][1])), color=(0,255,100), thickness=1)

                alt_view_cache[view][True] = np.copy(out)

                return alt_view_cache[view][show_objects]

            def render_side_view():
                
                #side view is a camera off to the right, looking left
                side_cam = np.array([[0, 0, -1, 2], [0, 1, 0, 0], [1, 0, 0, 2], [0, 0, 0, 1]])

                return render_alt_view(side_cam, "side")

            def render_top_view():

                #top view is a camera up above
                top_cam = np.array([[1, 0, 0, 0], [0, 0, 1, -2], [0, -1, 0, 1.5], [0, 0, 0, 1]])

                return render_alt_view(top_cam, "top")

            def render_alt_views():  
                cv2.imshow("side view", render_side_view())
                cv2.imshow("top view", render_top_view())

            print("frame: {0}".format(frame_ids[frame_ids_idx]))
            
            cv2.namedWindow("rendered frame")
            cv2.imshow("rendered frame", render_current_view())

            cv2.namedWindow("side view")
            cv2.namedWindow("top view")

            render_alt_views()

            def update_objects():
                nonlocal current_pose
                nonlocal object_meshes_and_bbs

                frame_pose = synchronized_poses_refined[frame_ids[frame_ids_idx]]
                sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ frame_pose
                new_current_pose = invert_affine(sensor_pose_in_annotated_coordinates)

                for obj_id in object_meshes_and_bbs.keys():
                    mesh = object_meshes_and_bbs[obj_id][0].transform(new_current_pose @ invert_affine(current_pose))
                    bb = apply_affine_to_points(apply_affine_to_points(object_meshes_and_bbs[obj_id][1], invert_affine(current_pose)), new_current_pose)
                    object_meshes_and_bbs[obj_id] = (mesh, bb)

                current_pose = new_current_pose

                cv2.imshow("rendered frame", render_current_view())

            def toggle_vis():
                nonlocal show_objects
                show_objects = not show_objects
                update_objects()

    #------------------------------------------------------------------------------------------
            def increment_frame_id():
                nonlocal frame_ids_idx
                nonlocal alt_view_cache

                #clear alt_view_cache for new frame (rerender)
                alt_view_cache = {}

                frame_ids_idx += 1
                frame_ids_idx = frame_ids_idx % len(frame_ids)

                synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] = synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] @ current_refinement[:3,:3]
                synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,3] += current_refinement[:3,3]

                print("frame: {0}".format(frame_ids[frame_ids_idx]))
            
                update_objects()

    #------------------------------------------------------------------------------------------

            #ROTATION STUFF

            min_rotation_delta = 0.001
            max_rotation_delta = 0.01
            rotation_velocity = 0 #from 0 -> 1
            rotation_velocity_delta = 0.15

            rotation_delta = min_rotation_delta
            last_rot_type = ""

            def rotate_using_euler(euler):
                nonlocal current_refinement
                nonlocal synchronized_poses_refined

                delta_rot_mat = R.from_euler("XYZ", euler).as_matrix()

                synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] = delta_rot_mat @ synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] @ delta_rot_mat.T
                current_refinement[:3,:3] = current_refinement[:3,:3] @ delta_rot_mat.T

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
                nonlocal current_refinement
                nonlocal synchronized_poses_refined

                synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,3] += synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] @ trans
                current_refinement[:3,3] += synchronized_poses_refined[frame_ids[frame_ids_idx]][:3,:3] @ trans

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
                    synchronized_poses_refined[frame_ids[frame_ids_idx]] = np.copy(synchronized_poses_corrected[frame_ids[frame_ids_idx]])
                    current_refinement = np.eye(4)
                elif k == ord('u'):
                    increase_rotation_alpha()
                elif k == ord('i'):
                    decrease_rotation_alpha()
                elif k == ord('p'):
                    increase_rotation_beta()
                elif k == ord('o'):
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
                elif k == ord('a'):
                    decrease_x()
                elif k == ord('d'):
                    increase_x()
                elif k == ord('1'):
                    toggle_vis()
                elif k == ord('2'):   
                    render_alt_views()

        # change of coordinates for synchronized poses from:
        # (sensor -> opti) back to (virtual -> opti)
        synchronized_poses_refined = {frame_id: sensor_pose @ camera_extrinsics for frame_id, sensor_pose in synchronized_poses_refined.items()}
        synchronized_poses_refined_df = convert_pose_dict_to_df(synchronized_poses_refined)

        if write_to_file:
            self.save_refined_poses_df(scene_dir, synchronized_poses_refined_df)

        return synchronized_poses_refined
