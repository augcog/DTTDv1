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
from datasets.ycb.dataset import YCBObjectPoints as SegDataset_ycb
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
opt = parser.parse_args()

cfg = Config()

def main():
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

    with torch.no_grad():

        for now, end_points in enumerate(test_dataloader):

            print("obj idx: {0}".format(now))

            torch.cuda.empty_cache()

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


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()
