"""
Understand which points are the most important for regression output
"""

import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
import cv2
from lib.randla_utils import randla_processing
from cfg.config import YCBConfig as Config, load_config

from lib.loss import Loss

try:
    from lib.tools import compute_rotation_matrix_from_ortho6d
except:
    from tools import compute_rotation_matrix_from_ortho6d

import open3d as o3d

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--config', type=str, default='', help='load a saved config')
parser.add_argument('--output', type=str, default='saliency', help='output for point vis')
parser.add_argument('--use_posecnn_rois', action="store_true", default=False, help="use the posecnn roi's")
opt = parser.parse_args()


cfg = Config()
if opt.config:
    cfg = load_config(cfg, opt.config)

cfg.refine_start = opt.refine_model != ''

if opt.use_posecnn_rois:
    from datasets.ycb.dataset import PoseDatasetPoseCNNResults as PoseDataset
else:
    from datasets.ycb.dataset import PoseDatasetAllObjects as PoseDataset

batch_size = 1
workers = 1
cfg.posecnn_results = "YCB_Video_toolbox/results_PoseCNN_RSS2018"

def main():
    if not os.path.isdir(opt.output):
        os.mkdir(opt.output)

    estimator = PoseNet(cfg = cfg)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()

    
    if opt.refine_model != '':
        refiner = PoseRefineNet(cfg = cfg)
        refiner.cuda()
        refiner.load_state_dict(torch.load(opt.refine_model))
        refiner.eval()

    test_dataset = PoseDataset('test', cfg = cfg)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    colors = [(96, 60, 20), (156, 39, 6), (212, 91, 18), (243, 188, 46), (95, 84, 38)]
    for now, data_objs in enumerate(test_dataloader):

        print("frame: {0}".format(now))

        color_img_file = '{0}/{1}-color.png'.format(cfg.root, test_dataset.list[now])
        color_img = cv2.imread(color_img_file)

        for obj_idx, end_points in enumerate(data_objs):

            torch.cuda.empty_cache()

            intr = end_points["intr"]
            del end_points["intr"]

            end_points_cuda = {}
            for k, v in end_points.items():
                end_points_cuda[k] = Variable(v).cuda()

            end_points = end_points_cuda

            if cfg.pcld_encoder == "randlanet":
                end_points = randla_processing(end_points, cfg)

            cam_fx, cam_fy, cam_cx, cam_cy = [x.item() for x in intr]
                                                
            #compute saliency
            #===========================================================================

            cloud = end_points["cloud"].clone().detach().requires_grad_(True)
            cloud_mean = end_points["cloud_mean"].clone().detach().requires_grad_(True)

            if cfg.use_normals:
                normals = end_points["normals"].clone().detach().requires_grad_(True)

            if cfg.use_colors:
                cloud_colors = end_points["cloud_colors"].clone().detach().requires_grad_(True)

            img = end_points["img"].clone().detach().requires_grad_(True)

            choose = end_points["choose"].clone().detach()

            end_points_test = {}
            end_points_test["cloud"] = cloud
            end_points_test["cloud_mean"] = cloud_mean
            if cfg.use_normals:
                end_points_test["normals"] = normals
            if cfg.use_colors:
                end_points_test["cloud_colors"] = cloud_colors        
            end_points_test["img"] = img
            end_points_test["choose"] = choose

            for k, v in end_points.items():
                if k not in end_points_test.keys():
                    end_points_test[k] = v

            end_points_test = estimator(end_points_test)

            o_i = end_points_test["obj_idx"].cpu().detach().item()

            pred_r = end_points_test["pred_r"]
            pred_t = end_points_test["pred_t"]

            if cfg.use_confidence:
                pred_c = end_points_test["pred_c"]
                #examine top 10 confidences
                k = 10

                pred_c = pred_c.view(1, cfg.num_points)
                top_cs, which_max = torch.topk(pred_c, k=k, dim=1)
                pred_t = pred_t.view(1 * cfg.num_points, 3)

                my_c = pred_c[0][which_max[0]]
                my_r = pred_r[0][which_max[0]]
                my_t = pred_t[which_max[0]]
            else:
                my_r = pred_r.squeeze()[1000]
                my_t = pred_t.squeeze()[1000]

            if cfg.use_confidence:
                dcloud_c, dimg_c = torch.autograd.grad(my_c, (cloud, img), torch.ones(my_c.shape).cuda(), retain_graph=True)
            dcloud_r, dimg_r = torch.autograd.grad(my_r, (cloud, img), torch.ones(my_r.shape).cuda(), retain_graph=True)
            dcloud_t, dimg_t = torch.autograd.grad(my_t, (cloud, img), torch.ones(my_t.shape).cuda())

            if cfg.use_confidence:
                dimg_c, _ = torch.max(torch.abs(dimg_c), dim=1)
                dimg_c = dimg_c.cpu().detach().numpy().squeeze()
                dimg_out_c = (dimg_c / np.max(dimg_c) * 65535).astype(np.uint16)
                output_filename = '{0}/{1}_{2}_dimg_c_saliency.png'.format(opt.output, now, o_i)
                cv2.imwrite(output_filename, dimg_out_c)

            dimg_t, _ = torch.max(torch.abs(dimg_t), dim=1)
            dimg_t = dimg_t.cpu().detach().numpy().squeeze()
            dimg_out_t = (dimg_t / np.max(dimg_t) * 65535).astype(np.uint16)
            output_filename = '{0}/{1}_{2}_dimg_t_saliency.png'.format(opt.output, now, o_i)
            cv2.imwrite(output_filename, dimg_out_t)

            dimg_r, _ = torch.max(torch.abs(dimg_r), dim=1)
            dimg_r = dimg_r.cpu().detach().numpy().squeeze()
            dimg_out_r = (dimg_r / np.max(dimg_r) * 65535).astype(np.uint16)
            output_filename = '{0}/{1}_{2}_dimg_r_saliency.png'.format(opt.output, now, o_i)
            cv2.imwrite(output_filename, dimg_out_r)

            if cfg.use_confidence:
                dcloud_c, _ = torch.max(torch.abs(dcloud_c), dim=2)
                dcloud_c = dcloud_c.cpu().detach().numpy().squeeze()

            dcloud_r, _ = torch.max(torch.abs(dcloud_r), dim=2)
            dcloud_r = dcloud_r.cpu().detach().numpy().squeeze()

            dcloud_t, _ = torch.max(torch.abs(dcloud_t), dim=2)
            dcloud_t = dcloud_t.cpu().detach().numpy().squeeze()

            choose = choose.cpu().detach().numpy().squeeze()

            if cfg.use_confidence:
                dcloud_out_c = np.zeros_like(dimg_out_c).flatten()
                dcloud_out_c[choose] = (dcloud_c / np.max(dcloud_c) * 65535).astype(np.uint16)
                dcloud_out_c = dcloud_out_c.reshape(dimg_out_c.shape)
                output_filename_c = '{0}/{1}_{2}_dcloud_c_saliency.png'.format(opt.output, now, o_i)
                cv2.imwrite(output_filename_c, dcloud_out_c)

            dcloud_out_r = np.zeros_like(dimg_out_r).flatten()
            dcloud_out_r[choose] = (dcloud_r / np.max(dcloud_r) * 65535).astype(np.uint16)
            dcloud_out_r = dcloud_out_r.reshape(dimg_out_r.shape)
            output_filename_r = '{0}/{1}_{2}_dcloud_r_saliency.png'.format(opt.output, now, o_i)
            cv2.imwrite(output_filename_r, dcloud_out_r)

            dcloud_out_t = np.zeros_like(dimg_out_t).flatten()
            dcloud_out_t[choose] = (dcloud_t / np.max(dcloud_t) * 65535).astype(np.uint16)
            dcloud_out_t = dcloud_out_t.reshape(dimg_out_t.shape)
            output_filename_t = '{0}/{1}_{2}_dcloud_t_saliency.png'.format(opt.output, now, o_i)
            cv2.imwrite(output_filename_t, dcloud_out_t)

            output_filename_cropped = '{0}/{1}_{2}_color_cropped.png'.format(opt.output, now, o_i)
            color_cropped = end_points_test["cropped_image"].cpu().detach().numpy().squeeze()
            cv2.imwrite(output_filename_cropped, color_cropped)


            #===========================================================================

if __name__ == "__main__":
    main()
