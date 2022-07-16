# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

from distutils.command.config import config
from tracemalloc import start
import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset import YCBSemanticSegDatasetWithIntr as SegDataset_ycb
from lib.randlanet import SegNet
from lib.loss import Loss
from lib.utils import setup_logger
from lib.randla_utils import randla_processing
from cfg.config import YCBConfig as Config, write_config
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ExponentialLR
import cv2

from tqdm import tqdm
import open3d as o3d
from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='Randlanet SegNet model')
parser.add_argument('--output', type=str, default="saliency")
opt = parser.parse_args()

cfg = Config()

def project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy):
    proj_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
    projected_pts = pts @ proj_mat.T
    projected_pts /= np.expand_dims(projected_pts[:,2], -1)
    projected_pts = projected_pts[:,:2]
    return projected_pts

def main():

    if not os.path.isdir(opt.output):
        os.mkdir(opt.output)

    cfg.manualSeed = 2023
    random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)
    np.random.seed(cfg.manualSeed)

    estimator = SegNet(cfg = cfg)
    estimator.cuda()
    estimator.load_state_dict(torch.load('{0}'.format(opt.model)))

    workers=1
    bs = 1

    optimizer = optim.Adam(estimator.parameters(), lr=cfg.lr)
    test_dataset = SegDataset_ycb('test', cfg = cfg)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=workers)

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the testing set: {0}\n'.format(len(test_dataset)))

    criterion = Loss()

    test_acc_avg = 0.0
    test_count = 0
    estimator.eval()

    for now, end_points in enumerate(test_dataloader):

        print("frame: {0}".format(now))

        color_img_file = '{0}/{1}-color.png'.format(cfg.root, test_dataset.list[now])
        color_img = cv2.imread(color_img_file)

        h, w, _ = color_img.shape

        torch.cuda.empty_cache()

        intr = end_points["intr"]
        del end_points["intr"]

        end_points_cuda = {}
        for k, v in end_points.items():
            end_points_cuda[k] = Variable(v).cuda()
        end_points_cuda["cloud"] = Variable(end_points_cuda["cloud"], requires_grad=True)
        end_points_cuda["normals"] = Variable(end_points_cuda["normals"], requires_grad=True)
        end_points_cuda["cloud_colors"] = Variable(end_points_cuda["cloud_colors"], requires_grad=True)

        end_points = end_points_cuda

        end_points = randla_processing(end_points, cfg)

        end_points = estimator(end_points)

        loss, acc = criterion(end_points)

        test_acc_avg += acc.item()
        test_count += 1

        print("current test pixel acc: {0}".format(test_acc_avg / test_count))

        pts = end_points["cloud"] + end_points["cloud_mean"]
        pts = pts.squeeze(0).detach().cpu().numpy()
        cam_fx, cam_fy, cam_cx, cam_cy = [x.item() for x in intr]
        projected_pts = project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy)
        normals = end_points["normals"]
        cloud_colors = end_points["cloud_colors"]

        choose = end_points["choose"].clone().detach()
        choose = choose.cpu().detach().numpy().squeeze()

        for i in range(21):
            preds = end_points["preds"]
            mask = preds == (i + 1)

            if torch.count_nonzero(mask) < 30:
                continue

            pred_logits = torch.transpose(end_points["pred_seg"][0], 0, 1)

            pred_logits = pred_logits[mask[0]][:, i + 1]

            print(i, pred_logits.shape)

            dcloud, dnormals, dcolors = torch.autograd.grad(pred_logits, (end_points["cloud"],  normals, cloud_colors), torch.ones(pred_logits.shape).cuda(), retain_graph=True)

            dcloud = dcloud.squeeze(0)
            dnormals = dnormals.squeeze(0)
            dcolors = dcolors.squeeze(0)

            dcloud, _ = torch.max(torch.abs(dcloud), dim=1)
            dnormals, _ = torch.max(torch.abs(dnormals), dim=1)
            dcolors, _ = torch.max(torch.abs(dcolors), dim=1)

            dcloud = dcloud.cpu().detach().numpy()
            dnormals = dnormals.cpu().detach().numpy()
            dcolors = dcolors.cpu().detach().numpy()

            doutcloud = np.zeros((h, w)).astype(np.float32).flatten()
            doutnormals = np.zeros((h, w)).astype(np.float32).flatten()
            doutcolors = np.zeros((h, w)).astype(np.float32).flatten()

            dcloud = dcloud / np.max(dcloud)
            dnormals = dnormals / np.max(dnormals)
            dcolors = dcolors / np.max(dcolors)

            dcloud[dcloud < 0.9] = 0
            dnormals[dnormals < 0.9] = 0
            dcolors[dcolors < 0.9] = 0

            doutcloud[choose] = (dcloud * 65535).astype(np.uint16)
            doutnormals[choose] = (dnormals * 65535).astype(np.uint16)
            doutcolors[choose] = (dcolors * 65535).astype(np.uint16)

            doutcloud = doutcloud.reshape((h, w))
            doutnormals = doutnormals.reshape((h, w))
            doutcolors = doutcolors.reshape((h, w))

            out_cloud_name = '{0}/{1}_{2}_dcloud_saliency.png'.format(opt.output, now, i)
            cv2.imwrite(out_cloud_name, doutcloud)

            out_normals_name = '{0}/{1}_{2}_dnormals_saliency.png'.format(opt.output, now, i)
            cv2.imwrite(out_normals_name, doutnormals)

            out_colors_name = '{0}/{1}_{2}_dcolors_saliency.png'.format(opt.output, now, i)
            cv2.imwrite(out_colors_name, doutcolors)

        if now > 10:
            break


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()
