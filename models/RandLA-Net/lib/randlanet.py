import argparse
import os
import random
from sympy import E
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from lib.RandLA.RandLANet import Network as RandLANet
import lib.pytorch_utils as pt_utils

class SegNet(nn.Module):
    def __init__(self, cfg):
        super(SegNet, self).__init__()
        self.num_points = cfg.num_points

        self.num_obj = cfg.num_objects
        
        self.rndla = RandLANet(cfg=cfg)

        self.cfg = cfg

    def forward(self, end_points):
        features = end_points["cloud"]
        bs = features.shape[0]

        normals = end_points["normals"]
        colors = end_points["cloud_colors"]

        features = torch.cat((features, normals, colors), dim=-1)

        end_points["RLA_features"] = features.transpose(1, 2)
        end_points = self.rndla(end_points)
        feat_x = end_points["RLA_embeddings"]
        logits = end_points["RLA_logits"]

        end_points["pred_seg"] = logits

        return end_points