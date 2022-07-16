import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
from lib.transformations import rotation_matrix_of_axis_angle
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import open3d as o3d
from lib.depth_utils import compute_normals, fill_missing
import cv2
import torch.nn.functional as F
import json
from .data_utils import load_cameras_dir, load_data_list, load_objects_dir, load_scene_metas

def get_random_rotation_around_symmetry_axis(axis, symm_type, num_symm):
    if symm_type == "radial":
        if num_symm == "inf":
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            angles = np.arange(0, 2 * np.pi, 2 * np.pi / int(num_symm))
            angle = np.random.choice(angles)
        return rotation_matrix_of_axis_angle(axis, angle).squeeze()
    else:
        raise Exception("Invalid symm_type " + symm_type)

def project_depth(depth, depth_scale, intrinsic, distortion, prune_zero=True):

    points = np.array([[(k, i) for k in range(depth.shape[1])] for i in range(depth.shape[0])]).reshape((-1, 2)).astype(np.float32)
    Z = depth.flatten() * depth_scale

    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]

    if prune_zero:
        points = points[Z > 0]
        Z = Z[Z > 0]

    # Step 1. Undistort.
    points_undistorted = cv2.undistortPoints(np.expand_dims(points, 1), intrinsic, distortion, P=intrinsic)
    points_undistorted = np.squeeze(points_undistorted, axis=1)

    # Step 2. Reproject.
    pts_xyz = np.zeros((points_undistorted.shape[0], 3))

    pts_xyz[:,0] = (points_undistorted[:, 0] - c_x) / f_x * Z
    pts_xyz[:,1] = (points_undistorted[:, 1] - c_y) / f_y * Z
    pts_xyz[:,2] = Z

    return pts_xyz

class SegDataset(data.Dataset):
    def __init__(self, mode, cfg):

        self.cfg = cfg

        cameras_dir = os.path.join(cfg.root, "cameras")
        self.data_dir = os.path.join(cfg.root, "data")
        objects_dir = os.path.join(cfg.root, "objects")

        self.cameras = load_cameras_dir(cameras_dir)
        self.objects = load_objects_dir(objects_dir)
        self.scene_metadatas = load_scene_metas(self.data_dir)

        if mode == "train":
            data_list_file = os.path.join(cfg.root, "train_data_list.txt")
            add_noise = True
        else:
            data_list_file = os.path.join(cfg.root, "test_data_list.txt")
            add_noise = False

        self.data_list = load_data_list(data_list_file)

        self.add_noise = add_noise
        self.length = len(self.data_list)

        self.mode = mode
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

    def __getitem__(self, index):
        item = self.data_list[index]
        item_abs = os.path.join(self.data_dir, item)
        scene_name = item[:item.find("/")]

        rgb = Image.open(item_abs + "_color.png")
        depth = np.array(Image.open(item_abs + "_depth.png"))
        label = np.array(Image.open(item_abs + "_label.png"))
        meta_file = item_abs + "_meta.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)

        scene_metadata = self.scene_metadatas[scene_name]

        cam_scale = scene_metadata["cam_scale"]
        camera = scene_metadata["camera"]
        objects = scene_metadata["objects"]

        camera_data = self.cameras[camera]
        camera_intr = camera_data["intrinsic"]
        camera_dist = camera_data["distortion"]

        valid_depth_mask = depth > 0

        mask = valid_depth_mask.flatten()

        choose = mask.nonzero()[0]

        if len(choose) > self.cfg.num_points:
            choose = np.random.choice(choose, self.cfg.num_points, replace=False)
        else:
            choose = np.pad(choose, (0, self.cfg.num_points - len(choose)), 'wrap')

        pcld = project_depth(depth, cam_scale, camera_intr, camera_dist, prune_zero=False)

        if self.add_noise:
            rgb = self.trancolor(rgb)

        rgb = np.array(rgb)
        #normalize RGB
        rgb = ((rgb.astype(np.float32) / 255.) - 0.5) * 2.

        pcld = pcld[choose]

        pcld_mean = np.mean(pcld, axis=0)

        pcld -= pcld_mean

        pcld_rgb = rgb.reshape((-1, 3))[choose]
        label = label.flatten()[choose]

        object_poses = meta["object_poses"]
        object_poses = {int(k): v for k, v in object_poses.items()}

        if self.cfg.use_normals:
            depth_mm = (depth * 1000 * cam_scale).astype(np.uint16)
            normals = compute_normals(depth_mm, camera_intr[0,0], camera_intr[1,1])
            normals_masked = normals.reshape((-1, 3))[choose].astype(np.float32)

        end_points = {}
        end_points["choose"] = torch.from_numpy(choose.astype(np.int32))
        end_points["img"] = torch.from_numpy(rgb.astype(np.float32))
        end_points["cloud"] = torch.from_numpy(pcld.astype(np.float32))
        end_points["cloud_mean"] = torch.from_numpy(pcld_mean.astype(np.float32))
        if self.cfg.use_normals:
            end_points["normals"] = torch.from_numpy(normals_masked.astype(np.float32))
        if self.cfg.use_colors:
            end_points["cloud_colors"] = torch.from_numpy(pcld_rgb.astype(np.float32))
        end_points["gt_seg"] = torch.from_numpy(label.astype(np.int64))

        return end_points

    def __len__(self):
        return self.length