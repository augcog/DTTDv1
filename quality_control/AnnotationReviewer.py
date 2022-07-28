import cv2
import matplotlib.pyplot as plt
import numpy as np
from regex import E
import pyrender
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import yaml


import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.affine_utils import invert_affine
from utils.camera_utils import load_intrinsics
from utils.constants import AZURE_KINECT_COLOR_HEIGHT, AZURE_KINECT_COLOR_WIDTH
from utils.frame_utils import load_bgr, load_label, load_meta

class AnnotationReviewer():
    def __init__(self, trimesh_objects={}, number_of_points=10000):
        self._objects = {}
        for obj_id, obj_data in trimesh_objects.items():
            self._objects[obj_id] = obj_data["mesh"].copy()

    def review_scene_annotations(self, scene_dir):

        frames_dir = os.path.join(scene_dir, "data")

        scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")

        with open(scene_metadata_file, 'r') as file:
            scene_metadata = yaml.safe_load(file)

        num_frames = scene_metadata["num_frames"]

        y_flip = R.from_euler("xyz", np.array([0, 0, np.pi])).as_matrix()
        z_flip = R.from_euler("xyz", np.array([0, np.pi, 0])).as_matrix()

        obj_nodes = {}

        # compose scene
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
        camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

        light_pose = np.eye(4)
        light_pose[:3,:3] = z_flip @ light_pose[:3,:3]
        scene.add(light, pose=light_pose)

        cam_pose = np.eye(4)
        cam_pose[:3,:3] = y_flip @ z_flip @ cam_pose[:3,:3]
        scene.add(camera, pose=cam_pose)

        # render scene
        r = pyrender.OffscreenRenderer(AZURE_KINECT_COLOR_WIDTH, AZURE_KINECT_COLOR_HEIGHT)

        for obj_id, obj_mesh in self._objects.items():
            mesh = pyrender.Mesh.from_trimesh(obj_mesh, smooth=False)
            obj_nodes[obj_id] = scene.add(mesh, pose=np.eye(4))

        rendered_frames = []

        for frame_id in tqdm(range(num_frames), total=num_frames, desc="rendering objects"):

            frame_meta = load_meta(frames_dir, frame_id)
            transforms = frame_meta["object_poses"]
            transforms = {int(k) : np.array(v) for k, v in transforms.items()}

            for obj_id, transform in transforms.items():
                scene.set_pose(obj_nodes[obj_id], transform)
            
            color, _ = r.render(scene)
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            rendered_frames.append(color)

        overlayed_label_frames = []

        colors = np.array([(0, 0, 0), (51, 102, 255), (153, 51, 255), (204, 0, 204), (255, 204, 0), (153, 204, 0)
            , (0, 102, 102), (51, 102, 0), (153, 0, 204), (102, 0, 51), (102, 255, 255), (102, 255, 153)
            , (153, 51, 0), (102, 153, 153), (102, 51, 0), (153, 153, 102), (255, 204, 153), (255, 102, 102), (0, 255, 153)
            , (102, 0, 102), (153, 255, 51), (51, 102, 153)])

        colors = np.tile(np.expand_dims(np.expand_dims(colors, 0), 0), (AZURE_KINECT_COLOR_HEIGHT, AZURE_KINECT_COLOR_WIDTH, 1, 1))

        opacity = 0.7

        for frame_id in tqdm(range(num_frames), total=num_frames, desc="overlaying semantic segmentation"):

            color = load_bgr(frames_dir, frame_id, "jpg")
            label = np.expand_dims(np.expand_dims(load_label(frames_dir, frame_id), -1), -1)
            label_colors = np.take_along_axis(colors, label, axis=2).squeeze(2)

            color = color.astype(np.float32) / 255.
            label_colors = label_colors.astype(np.float32) / 255. * opacity

            color += label_colors
            color = np.clip(color, 0, 1)

            color = (color * 255.).astype(np.uint8)

            overlayed_label_frames.append(color)

        h, w = int(AZURE_KINECT_COLOR_HEIGHT / 2), int(AZURE_KINECT_COLOR_WIDTH / 2)

        while True:
            cv2.namedWindow("rendered", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("rendered", w, h)
            cv2.namedWindow("overlayed label", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("overlayed label", w, h)

            for rendered_frame, overlayed_label_frame in tqdm(zip(rendered_frames, overlayed_label_frames), total=num_frames, desc="final vis"):
                cv2.imshow("rendered", cv2.resize(rendered_frame, (w, h)))
                cv2.imshow("overlayed label", cv2.resize(overlayed_label_frame, (w, h)))
                cv2.waitKey(50)

            cv2.destroyWindow("rendered")
            cv2.destroyWindow("overlayed label")

            print("Go through scene again? (y/n)")
            if input().lower() == "y":
                continue
            else:
                break
            

    