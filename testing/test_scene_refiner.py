import argparse
import open3d as o3d
import numpy as np
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from data_processing import CameraPoseSynchronizer
from pose_refinement import ScenePoseRefiner
from utils.affine_utils import invert_affine
from utils.camera_utils import load_extrinsics, load_distortion, load_frame_intrinsics
from utils.constants import SCENES_DIR
from utils.frame_utils import load_rgb, load_depth
from utils.object_utils import load_object_meshes
from utils.pointcloud_utils import pointcloud_from_rgb_depth
from utils.pose_dataframe_utils import convert_pose_df_to_dict

def main():
    parser = argparse.ArgumentParser(description='Generate semantic labeling and meta labeling')
    parser.add_argument('scene_name', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')
    parser.set_defaults(refine=True)

    args = parser.parse_args()

    scene_dir = os.path.join(SCENES_DIR, args.scene_name)

    scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")
    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)
    
    objects = load_object_meshes(scene_metadata["objects"])

    if args.refine and len(objects) == 1:
        print("WARNING!, ICP refinement with only 1 object is dangerous. May result in bad ICP result.")

    annotated_poses_csv = os.path.join(scene_dir, "annotated_object_poses", "annotated_object_poses.yaml")
    with open(annotated_poses_csv, "r") as file:
        annotated_poses_data = yaml.safe_load(file)

    annotated_poses_single_frameid = annotated_poses_data["frame"]
    annotated_poses_single_frame = annotated_poses_data["object_poses"]
    annotated_poses_single_frame = {k: np.array(v) for k, v in annotated_poses_single_frame.items()}

    cam_pose_sync = CameraPoseSynchronizer()
    synchronized_poses_csv = os.path.join(scene_dir, "camera_poses", "camera_poses_synchronized.csv")
    synchronized_poses = cam_pose_sync.load_from_file(synchronized_poses_csv)
    synchronized_poses = convert_pose_df_to_dict(synchronized_poses)

    scene_pose_refiner = ScenePoseRefiner(objects)

    frames_dir = os.path.join(scene_dir, "data")

    scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")

    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)

    camera_name = scene_metadata["camera"]

    camera_intrinsics_dict = load_frame_intrinsics(scene_dir, raw=False)
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
    for obj_id, obj in scene_pose_refiner._objects.items():
        annotated_obj_pose = annotated_poses_single_frame[obj_id]
        obj_pcld = obj["pcld"].transform(annotated_obj_pose)
        object_pcld_transformed[obj_id] = obj_pcld

    synchronized_poses_refined = {}

    frame_id = 0

    sensor_pose = synchronized_poses_corrected[frame_id]
    rgb = load_rgb(frames_dir, frame_id, "jpg")
    depth = load_depth(frames_dir, frame_id)
    h, w, _ = rgb.shape

    sensor_pose_in_annotated_coordinates = sensor_pose_annotated_frame_inv @ sensor_pose
    sensor_pose_in_annotated_coordinates_inv = invert_affine(sensor_pose_in_annotated_coordinates)

    # First, refine pose
    objects_in_sensor_coords = {}

    for idx, (obj_id, obj_pcld) in enumerate(object_pcld_transformed.items()):
        obj_pcld = o3d.geometry.PointCloud(obj_pcld) #copy constructor
        obj_pcld = obj_pcld.transform(sensor_pose_in_annotated_coordinates_inv)
        objects_in_sensor_coords[obj_id] = obj_pcld
    
    camera_pcld = pointcloud_from_rgb_depth(rgb, depth, cam_scale, camera_intrinsics_dict[frame_id], camera_distortion)

    pose_refinement_icp = scene_pose_refiner.refine_pose_icp(list(objects_in_sensor_coords.values()), camera_pcld)

    synchronized_poses_refined[frame_id] = sensor_pose @ invert_affine(pose_refinement_icp)

    #visualize
    orig_pose = synchronized_poses_corrected[frame_id]
    refined_pose = synchronized_poses_refined[frame_id]

    camera_pcld_filename = os.path.join(dir_path, "camera_pcld.ply")
    orig_objs_filename = os.path.join(dir_path, "orig_objs.ply")
    refined_objs_filename = os.path.join(dir_path, "refined_objs.ply")

    obj_points = []

    for idx, (obj_id, obj_pcld) in enumerate(object_pcld_transformed.items()):
        obj_pcld = o3d.geometry.PointCloud(obj_pcld) #copy constructor
        obj_points.append(np.array(obj_pcld.points))

    obj_points = np.array(obj_points)
    obj_points = obj_points.reshape((-1, 3))

    orig_pcld = o3d.geometry.PointCloud()
    orig_pcld.points = o3d.utility.Vector3dVector(obj_points)

    refined_pcld = o3d.geometry.PointCloud(orig_pcld)

    orig_transform = invert_affine(sensor_pose_annotated_frame_inv @ orig_pose)
    orig_pcld = orig_pcld.transform(orig_transform)

    refined_transform = invert_affine(sensor_pose_annotated_frame_inv @ refined_pose)
    refined_pcld = refined_pcld.transform(refined_transform)

    o3d.io.write_point_cloud(camera_pcld_filename, camera_pcld)
    o3d.io.write_point_cloud(orig_objs_filename, orig_pcld)
    o3d.io.write_point_cloud(refined_objs_filename, refined_pcld)

if __name__ == "__main__":
    main()