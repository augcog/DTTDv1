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
parser.add_argument('--output', type=str, default="segmentation")
opt = parser.parse_args()

cfg = Config()

def project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy):
    proj_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
    projected_pts = pts @ proj_mat.T
    projected_pts /= np.expand_dims(projected_pts[:,2], -1)
    projected_pts = projected_pts[:,:2]
    return projected_pts

colors = [(255, 255, 255), (51, 102, 255), (153, 51, 255), (204, 0, 204), (255, 204, 0), (153, 204, 0)
            , (0, 102, 102), (51, 102, 0), (153, 0, 204), (102, 0, 51), (102, 255, 255), (102, 255, 153)
            , (153, 51, 0), (102, 153, 153), (102, 51, 0), (153, 153, 102), (255, 204, 153), (255, 102, 102), (0, 255, 153)
            , (102, 0, 102), (153, 255, 51), (51, 102, 153)]

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
    test_dataset = SegDataset_ycb('train', cfg = cfg)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=workers)

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the testing set: {0}\n'.format(len(test_dataset)))

    criterion = Loss()

    test_acc_avg = 0.0
    test_count = 0
    estimator.eval()

    with torch.no_grad():

        for now, end_points in enumerate(test_dataloader):

            print("frame: {0}".format(now))

            color_img_file = '{0}/{1}-color.png'.format(cfg.root, test_dataset.list[now])
            color_img = cv2.imread(color_img_file)
            color_img_2 = cv2.imread(color_img_file)

            torch.cuda.empty_cache()

            intr = end_points["intr"]
            del end_points["intr"]

            end_points_cuda = {}
            for k, v in end_points.items():
                end_points_cuda[k] = Variable(v).cuda()

            end_points = end_points_cuda

            end_points = randla_processing(end_points, cfg)

            end_points = estimator(end_points)

            loss, acc = criterion(end_points)

            test_acc_avg += acc.item()
            test_count += 1

            print("current test pixel acc: {0}".format(test_acc_avg / test_count))

            preds = end_points["preds"].squeeze(0).detach().cpu().numpy()
            pts = end_points["cloud"] + end_points["cloud_mean"]
            pts = pts.squeeze(0).detach().cpu().numpy()
            cam_fx, cam_fy, cam_cx, cam_cy = [x.item() for x in intr]
            projected_pts = project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy)

            gt_seg = end_points["gt_seg"].squeeze().detach().cpu().numpy()
            
            for (x, y), c in zip(projected_pts, preds):
                r, g, b = colors[c]
                color_img = cv2.circle(color_img, (int(x), int(y)), radius=1, color=(b,g,r), thickness=-1)

            for (x, y), c in zip(projected_pts, gt_seg):
                r, g, b = colors[c]
                color_img_2 = cv2.circle(color_img_2, (int(x), int(y)), radius=1, color=(b,g,r), thickness=-1)
            
            output_filename = '{0}/{1}.png'.format(opt.output, now)
            cv2.imwrite(output_filename, color_img)

            output_filename_2 = '{0}/{1}_GT.png'.format(opt.output, now)
            cv2.imwrite(output_filename_2, color_img_2)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()
