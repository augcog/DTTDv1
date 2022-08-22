import cv2
import numpy as np
import open3d as o3d
import pyrender
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import trimesh 
import threading
import yaml

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.affine_utils import affine_matrix_from_rotmat_trans, invert_affine
from utils.frame_utils import write_rgb, write_depth, write_label, write_meta
from utils.object_utils import load_object_meshes, load_object_meshes_trimesh

class SyntheticDataGenerator:
    def __init__(self, object_ids = [], seed=2022):
        self.seed = seed
        self._objects = {}

        objects_trimesh = load_object_meshes_trimesh(object_ids)
        objects_o3d = load_object_meshes(object_ids)
                
        for obj_id in object_ids:
            obj_trimesh_data = objects_trimesh[obj_id]
            name = obj_trimesh_data['name']
            obj_trimesh = obj_trimesh_data['mesh']
            obj_o3d = objects_o3d[obj_id]['mesh']

            # Compute max radius
            pts = np.array(obj_o3d.vertices)
            extent = np.max(pts, axis=0) - np.min(pts, axis=0)
            max_radius = np.max(extent)


            # PointCloud for collision detection
            obj_o3d_collision = o3d.geometry.TriangleMesh(obj_o3d)
            obj_o3d_collision = obj_o3d_collision.simplify_quadric_decimation(3000)

            # Want to create a copy of the obj_trimesh that is colored with the label
            obj_trimesh_label_colored = obj_trimesh.copy()
            
            vertices = np.array(obj_trimesh_label_colored.vertices)

            obj_label = (np.ones_like(vertices) * obj_id).astype(np.uint8)

            trimesh_color = trimesh.visual.ColorVisuals(vertex_colors = obj_label)
            
            obj_trimesh_label_colored.visual = trimesh_color
            
            object_data = {}
            object_data['name'] = name
            object_data['trimesh'] = obj_trimesh
            object_data['o3d'] = obj_o3d
            object_data['label_trimesh'] = obj_trimesh_label_colored
            object_data['o3d_collision'] = obj_o3d_collision
            object_data['max_radius'] = max_radius
 
            self._objects[obj_id] = object_data

    @staticmethod
    def _write_data(data_dir, frame_id, color, depth, label, frame_metadata):
        write_rgb(data_dir, frame_id, color, "jpg")
        write_depth(data_dir, frame_id, depth)
        write_label(data_dir, frame_id, label)
        write_meta(data_dir, frame_id, frame_metadata)

    def generate_synthetic_scene(self, output_scene_dir, num_frames, intr_file=None, cam_width=1280, cam_height=720):
        assert(os.path.exists(output_scene_dir))

        # The two "true" outputs
        data_dir = os.path.join(output_scene_dir, "data")
        scene_meta_file = os.path.join(output_scene_dir, "scene_meta.yaml")

        os.mkdir(data_dir)

        if not intr_file:
            intr_file = os.path.join(dir_path, "default_synthetic_intr.txt")

        cam_intrinsic = np.loadtxt(intr_file)

        y_flip = R.from_euler("xyz", np.array([0, 0, np.pi])).as_matrix()
        z_flip = R.from_euler("xyz", np.array([0, np.pi, 0])).as_matrix()

        obj_nodes = {}

        behind_camera_pose = np.eye(4)
        behind_camera_pose[2,3] = -2.

        # compose scene
        scene_color = pyrender.Scene(ambient_light=[.4, .4, .4], bg_color=[0., 0., 0.])
        scene_label = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0., 0., 0.])
        camera = pyrender.IntrinsicsCamera(fx = cam_intrinsic[0,0], fy=cam_intrinsic[1,1], cx=cam_intrinsic[0,2], cy=cam_intrinsic[1,2])

        cam_pose = np.eye(4)
        cam_pose[:3,:3] = y_flip @ z_flip @ cam_pose[:3,:3]
        scene_color.add(camera, pose=cam_pose)
        scene_label.add(camera, pose=cam_pose)

        light_pose = np.eye(4)
        light_pose[:3,:3] = z_flip @ light_pose[:3,:3]

        # render scene
        r = pyrender.OffscreenRenderer(cam_width, cam_height)

        obj_ids = []

        # For label renderer
        label_node_map = {}

        for obj_id, obj_data in self._objects.items():
            obj_mesh = obj_data['trimesh']
            mesh = pyrender.Mesh.from_trimesh(obj_mesh, smooth=False)

            obj_label_mesh = obj_data['label_trimesh']
            label_mesh = pyrender.Mesh.from_trimesh(obj_label_mesh, smooth=False)

            mesh_node = scene_color.add(mesh, pose=behind_camera_pose)
            label_node = scene_label.add(label_mesh, pose=behind_camera_pose)

            label_node_map[label_node] = (np.ones(3) * obj_id).astype(np.uint8)

            obj_nodes[obj_id] = (mesh_node, label_node)
            obj_ids.append(obj_id)

        prob_single = 0.2 #Place only one object in scene
        prob_far = 0.5 #Place every object at least one extent away so collisions are impossible
        prob_cluttered = 0.3 #Place objects close and use collisions detection (slow)

        assert(prob_single + prob_far + prob_cluttered == 1)

        single_thresh = prob_single
        far_thresh = prob_single + prob_far

        mean_location = np.array([0, 0, 0.7]).reshape((3, 1))

        # Maximum deviation in any direction for an objects from mean_location
        max_deviation = 0.3
        max_deviation_tight = 0.07
        
        # For multi-object frames
        min_objects = 2
        max_objects = min(len(obj_ids), 4)
        num_objects = np.arange(min_objects, max_objects + 1)

        max_move_counter = 10

        frame_write_threads = []

        obj_poses = {}
        objects_collision = {}

        # initialize poses
        for obj_id in obj_ids:
            obj_poses[obj_id] = np.eye(4)
            objects_collision[obj_id] = o3d.geometry.TriangleMesh(self._objects[obj_id]['o3d_collision'])
        
        # random positions of objects and update their object meshes
        def randomize_positions(objects_in_frame):
            nonlocal obj_poses
            nonlocal objects_collision

            for obj_id in objects_in_frame:
                obj_rot = R.random().as_matrix()
                if frame_type < far_thresh:
                    deviation = max_deviation
                else:
                    deviation = max_deviation_tight
                obj_trans = mean_location + np.random.uniform(-deviation, deviation, size=(3,1))
                new_pose = affine_matrix_from_rotmat_trans(obj_rot, obj_trans)
                objects_collision[obj_id] = objects_collision[obj_id].transform(new_pose @ invert_affine(obj_poses[obj_id]))
                obj_poses[obj_id] = new_pose

        for frame_id in tqdm(range(num_frames), total=num_frames, desc="generating frames"):
            frame_type = np.random.uniform()
            if frame_type < single_thresh:
                objects_in_frame = np.random.choice(obj_ids, size=1)
            else:
                objects_in_frame = np.random.choice(obj_ids, size=np.random.choice(num_objects), replace=False)

            randomize_positions(objects_in_frame)

            # Collision stuff
            if frame_type >= single_thresh:

                move_counter = 0

                # Move objects extent away from one another
                if frame_type < far_thresh:
                    while True:
                        moved = False
                        for obj_id1 in objects_in_frame:
                            for obj_id2 in objects_in_frame:

                                if obj_id1 == obj_id2:
                                    continue

                                mesh1_center = obj_poses[obj_id1][:3,3]
                                mesh2_center = obj_poses[obj_id2][:3,3]
                                diff = mesh2_center - mesh1_center

                                mesh1_max_radius = self._objects[obj_id1]['max_radius']
                                mesh2_max_radius = self._objects[obj_id2]['max_radius']
                                max_dia = max(mesh1_max_radius, mesh2_max_radius) * 2

                                if np.linalg.norm(diff) < max_dia:

                                    mesh1_translation = -diff * (max_dia - np.linalg.norm(diff) + 0.005) / 2.
                                    mesh2_translation = diff * (max_dia - np.linalg.norm(diff) + 0.005) / 2.

                                    print("intersect far", obj_id1, obj_id2, np.linalg.norm(diff), mesh1_translation, mesh2_translation)

                                    obj_poses[obj_id1][:3,3] += mesh1_translation
                                    obj_poses[obj_id2][:3,3] += mesh2_translation

                                    moved = True

                        if not moved:
                            break

                        if move_counter > max_move_counter:
                            randomize_positions(objects_in_frame)

                # Move objects incrementally using open3d collision detection
                else:

                    # Move 2 centimeters at a time
                    movement_size = 0.03

                    while True:
                        moved = False
                        for obj_id1 in objects_in_frame:
                            for obj_id2 in objects_in_frame:

                                if obj_id1 == obj_id2:
                                    continue

                                mesh1 = objects_collision[obj_id1]
                                mesh2 = objects_collision[obj_id2]

                                if mesh1.is_intersecting(mesh2):

                                    mesh1_center = obj_poses[obj_id1][:3,3]
                                    mesh2_center = obj_poses[obj_id2][:3,3]
                                    diff = mesh2_center - mesh1_center

                                    diff /= np.linalg.norm(diff, keepdims=True)

                                    print("intersect close", obj_id1, obj_id2, np.linalg.norm(diff))

                                    mesh1_translation = -diff * movement_size / 2.
                                    mesh2_translation = diff * movement_size / 2.

                                    obj_poses[obj_id1][:3,3] += mesh1_translation
                                    obj_poses[obj_id2][:3,3] += mesh2_translation

                                    objects_collision[obj_id1] = mesh1.translate(mesh1_translation)
                                    objects_collision[obj_id2] = mesh2.translate(mesh2_translation)

                                    moved = True

                        if not moved:
                            break
                            
                        if move_counter > max_move_counter:
                            randomize_positions(objects_in_frame)

            # First, take color picture
            light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
            light_node = scene_color.add(light, light_pose)
            for obj_id in objects_in_frame:

                pose = obj_poses[obj_id]

                mesh_node, _ = obj_nodes[obj_id]
                
                scene_color.set_pose(mesh_node, pose)

            color, depth = r.render(scene_color)

            #convert from meters to millimeters and uint16
            depth = (depth.astype(np.float32) * 1000).astype(np.uint16)

            scene_color.remove_node(light_node)

            # Then, take label picture
            for obj_id in objects_in_frame:

                pose = obj_poses[obj_id]

                _, label_node = obj_nodes[obj_id]

                scene_label.set_pose(label_node, pose)

            render_flags = pyrender.RenderFlags.SEG

            label, _ = r.render(scene_label, render_flags, label_node_map)

            label = label[:,:,0].astype(np.uint16)

            objects_in_image = sorted(np.unique(label).tolist())

            frame_metadata = {}
            frame_metadata["objects"] = [x for x in objects_in_image if x != 0]
            frame_metadata["object_poses"] = {int(k): v.tolist() for k, v in obj_poses.items() if k in frame_metadata["objects"]}
            frame_metadata["intrinsic"] = cam_intrinsic.tolist()
            frame_metadata["distortion"] = None

            frame_write_thread = threading.Thread(target=SyntheticDataGenerator._write_data, args=(data_dir, frame_id, color, depth, label, frame_metadata))
            frame_write_thread.start()
            frame_write_threads.append(frame_write_thread)

            # Cleanup
            for obj_id, (mesh_node, label_node) in obj_nodes.items():
                scene_color.set_pose(mesh_node, behind_camera_pose)
                scene_label.set_pose(label_node, behind_camera_pose)

        # Finish all write threads
        for frame_write_thread in frame_write_threads:
            frame_write_thread.join()

        # Write scene metadata
        scene_metadata = {}
        scene_metadata["cam_scale"] = 0.001
        scene_metadata["camera"] = "simulated"
        scene_metadata["num_frames"] = num_frames
        scene_metadata["objects"] = sorted(list(self._objects.keys()))

        with open(scene_meta_file, "w") as f:
            yaml.dump(scene_metadata, f)

            

