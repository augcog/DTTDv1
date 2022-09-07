import cv2
import json
import numpy as np
import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data_utils import load_cameras_dir, load_data_list, load_objects_dir, load_scene_metas

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

class PoseDataset(data.Dataset):
    def __init__(self, root, mode, add_noise, num_points=38400):
        
        self.data_dir = os.path.join(root, "data")
        objects_dir = os.path.join(root, "objects")

        self.objects = load_objects_dir(objects_dir)
        self.scene_metadatas = load_scene_metas(self.data_dir)

        if mode == "train":
            data_list_file = os.path.join(root, "train_data_list.txt")
        else:
            data_list_file = os.path.join(root, "test_data_list.txt")

        self.data_list = load_data_list(data_list_file)

        self.add_noise = add_noise
        self.length = len(self.data_list)

        self.root = root
        self.mode = mode
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.num_points = num_points

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

        camera_intr = np.array(meta["intrinsic"])
        camera_dist = np.array(meta["distortion"])

        valid_depth_mask = depth > 0

        mask = valid_depth_mask.flatten()

        choose = mask.nonzero()[0]

        if len(choose) > self.num_points:
            choose = np.random.choice(choose, self.num_points, replace=False)
        else:
            choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

        pcld = project_depth(depth, cam_scale, camera_intr, camera_dist, prune_zero=False)

        if self.add_noise:
            rgb = self.trancolor(rgb)

        rgb = np.array(rgb)
        #normalize RGB
        rgb = ((rgb.astype(np.float32) / 255.) - 0.5) * 2.

        pcld = pcld[choose]
        pcld_rgb = rgb.reshape((-1, 3))[choose]
        label = label.flatten()[choose]

        object_poses = meta["object_poses"]
        object_poses = {int(k): v for k, v in object_poses.items()}

        end_points = {}
        end_points["choose"] = torch.from_numpy(choose.astype(np.int32))
        end_points["img"] = torch.from_numpy(rgb.astype(np.float32))
        end_points["pcld_xyz"] = torch.from_numpy(pcld.astype(np.float32))
        end_points["pcld_rgb"] = torch.from_numpy(pcld_rgb.astype(np.float32))
        end_points["label"] = torch.from_numpy(label.astype(np.int64))

        return end_points

    def __len__(self):
        return self.length

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(dir_path, "root")
    mode = "train"

    dataset = PoseDataset(root, mode, add_noise=True)
    print(dataset.__getitem__(0))

if __name__ == "__main__":
    main()

