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

class PoseDataset(data.Dataset):
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
            self.add_noise = True
        else:
            data_list_file = os.path.join(cfg.root, "test_data_list.txt")
            self.add_noise = False

        self.data_list = load_data_list(data_list_file)

        self.length = len(self.data_list)

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.front_num = 2

        print(len(self.data_list))

        self.symmetry_obj_idx = []

    def get_item(self, index, idx, obj_idx, img, depth, label, meta, cam_scale, cam_intr, cam_dist, return_intr=False, sample_model=True):

        if self.cfg.fill_depth:
            depth = fill_missing(depth, 1 / cam_scale, 1)

        if self.cfg.blur_depth:
            depth = cv2.GaussianBlur(depth,(3,3),cv2.BORDER_DEFAULT)

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        if self.add_noise and self.cfg.add_front_aug:
            for k in range(5):
                seed = random.choice(self.syn)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.cfg.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.cfg.root, seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                   continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, obj_idx))
        mask = mask_label * mask_depth
        if len(mask.nonzero()[0]) <= self.minimum_num_pt:
            return {}

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        h, w, _= np.array(img).shape
        rmin, rmax, cmin, cmax = max(0, rmin), min(h, rmax), max(0, cmin), min(w, cmax)
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) == 0:
            return {}

        if len(choose) > self.cfg.num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.cfg.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.cfg.num_points - len(choose)), 'wrap')

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3][rmin:rmax, cmin:cmax,:]

        img = np.transpose(img, (2, 0, 1))
        img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        object_poses = meta["object_poses"]
        object_poses = {int(k): v for k, v in object_poses.items()}

        target_r = np.array(object_poses[obj_idx])[:3,:3]
        target_t = np.array(object_poses[obj_idx])[:3,3]

        #TODO: currently, no front annotations
        front = np.array([[0, 0, 0]])

        if self.add_noise and self.cfg.symm_rotation_aug:
            #PERFORM SYMMETRY ROTATION AUGMENTATION
            #symmetries
            symm = self.symmd[obj_idx]

            #calculate other peaks based on size of symm
            if len(symm) > 0:
                symm_type, num_symm = symm[0]
                symmetry_augmentation = get_random_rotation_around_symmetry_axis(front, symm_type, num_symm)
                target_r = target_r @ symmetry_augmentation

        choose = np.array([choose])

        depth_crop_mask = np.zeros_like(depth).astype(bool)
        depth_crop_mask[rmin:rmax, cmin:cmax] = True
        depth_crop_mask = depth_crop_mask.flatten()

        cloud = project_depth(depth, cam_scale, cam_intr, cam_dist, prune_zero=False)

        cloud = cloud[depth_crop_mask][choose].squeeze(0)

        if self.add_noise and self.cfg.noise_trans > 0:
            add_t = np.random.uniform(-self.cfg.noise_trans, self.cfg.noise_trans, (self.cfg.num_points, 3))
            cloud = np.add(cloud, add_t)

        # colors = orig_img.reshape((-1, 3))

        # camera_pcld = o3d.geometry.PointCloud()
        # camera_pcld.points = o3d.utility.Vector3dVector(cloud)
        # camera_pcld.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.)
        # o3d.io.write_point_cloud("camera_pcld_{0}.ply".format(index), camera_pcld)

        #NORMALS
        if self.cfg.use_normals:
            depth_mm = (depth * 1000 * cam_scale).astype(np.uint16)
            normals = compute_normals(depth_mm, cam_intr[0,0], cam_intr[1,1])
            normals_masked = normals[rmin:rmax, cmin:cmax].reshape((-1, 3))[choose].astype(np.float32).squeeze(0)

        model_points = self.objects[obj_idx]['pcld']
        if sample_model:
            if self.cfg.refine_start:
                select_list = np.random.choice(len(model_points), self.num_pt_mesh_large, replace=False) # without replacement, so that it won't choice duplicate points
            else:
                select_list = np.random.choice(len(model_points), self.num_pt_mesh_small, replace=False) # without replacement, so that it won't choice duplicate points
            model_points = model_points[select_list]

        target = np.dot(model_points, target_r.T)
        target = np.add(target, target_t)

        # target_pcld = o3d.geometry.PointCloud()
        # target_pcld.points = o3d.utility.Vector3dVector(target)
        # o3d.io.write_point_cloud("target_pcld_{0}_{1}.ply".format(index, obj_idx), target_pcld)
            
        target_front = np.dot(front, target_r.T)
        target_front = np.add(target_front, target_t)

        #[0-1]
        img_normalized = img_masked.astype(np.float32) / 255.
        img_normalized = self.norm(torch.from_numpy(img_normalized))

        if self.cfg.use_colors:
            cloud_colors = img_normalized.view((3, -1)).transpose(0, 1)[choose]

        end_points = {}

        end_points["cloud_mean"] = torch.from_numpy(np.mean(cloud.astype(np.float32), axis=0, keepdims=True))
        end_points["cloud"] = torch.from_numpy(cloud.astype(np.float32)) - end_points["cloud_mean"]

        if self.cfg.use_normals:
            end_points["normals"] = torch.from_numpy(normals_masked.astype(np.float32))

        if self.cfg.use_colors:
            end_points["cloud_colors"] = cloud_colors

        end_points["choose"] = torch.LongTensor(choose.astype(np.int32))
        end_points["img"] = img_normalized
        end_points["target"] = torch.from_numpy(target.astype(np.float32))
        end_points["target_front"] = torch.from_numpy(target_front.astype(np.float32))
        end_points["model_points"] = torch.from_numpy(model_points.astype(np.float32))
        end_points["front"] = torch.from_numpy(front.astype(np.float32))
        end_points["obj_idx"] = torch.LongTensor([int(obj_idx) - 1])

        if return_intr:
            end_points["intr"] = (cam_intr, cam_dist)

        return end_points

    def __getitem__(self, index):

        item = self.data_list[index]
        item_abs = os.path.join(self.data_dir, item)
        scene_name = item[:item.find("/")]

        img = Image.open(item_abs + "_color.png")
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

        obj = np.array(meta['objects']).flatten().astype(np.int32)
        idxs = [i for i in range(len(obj))]
        np.random.shuffle(idxs)

        for idx in idxs:
            obj_idx = obj[idx]

            end_points = self.get_item(index, idx, obj_idx, img, depth, label, meta, cam_scale, camera_intr, camera_dist)

            if end_points:
                return end_points
            else:
                continue
        print("no valid obj with framae {0}".format(self.list[index]))
        return {}

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.cfg.refine_start:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    @staticmethod
    def custom_collate_fn(data):
        data = [d for d in data if len(d) > 0]

        imgs = [d["img"] for d in data]

        max_height = max([i.shape[1] for i in imgs])
        max_width = max([i.shape[2] for i in imgs])

        orig_img_heights = [i.shape[1] for i in imgs]
        orig_img_widths = [i.shape[2] for i in imgs]

        imgs = [F.pad(i, (0, max_width - i.shape[2], 0, max_height - i.shape[1])) for i in imgs]

        chooses = [d["choose"] for d in data]

        for d, img, choose, orig_img_height, orig_img_width in zip(data, imgs, chooses, orig_img_heights, orig_img_widths):
            d["img"] = img
            choose = (torch.floor(choose / orig_img_width) * max_width + choose % orig_img_width).type(torch.LongTensor)
            d["choose"] = choose

        return torch.utils.data.dataloader.default_collate(data)

# class PoseDatasetAllObjects(PoseDataset):
#     def __getitem__(self, index):

#         color_filename = '{0}/{1}-color.png'.format(self.cfg.root, self.list[index])

#         img = Image.open(color_filename)
#         depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.cfg.root, self.list[index])))
#         label = self.get_label(index)
#         meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.cfg.root, self.list[index]))

#         obj = meta['cls_indexes'].flatten().astype(np.int32)

#         data_output = []

#         orig_img = img
#         orig_depth = np.copy(depth)
#         orig_label = np.copy(label)

#         for idx in range(len(obj)):
#             obj_idx = obj[idx]
#             img = orig_img

#             end_points = self.get_item(index, idx, obj_idx, img, depth, label, meta, return_intr=True, sample_model=False)

#             if end_points:
#                 data_output.append(end_points)
#                 img = orig_img
#                 depth = orig_depth
#                 label = orig_label
#             else:
#                 print("WARNING, FAILURE TO PROCESS OBJ {0} in FRAME {1}".format(obj_idx, color_filename))
#                 continue

#         return data_output


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_height = 720
img_width = 1280

def get_bbox_posecnn(posecnn_rois, idx):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_height:
        delt = rmax - img_height
        rmax = img_height
        rmin -= delt
    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    return rmin, rmax, cmin, cmax

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_height:
        delt = rmax - img_height
        rmax = img_height
        rmin -= delt
    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    return rmin, rmax, cmin, cmax
