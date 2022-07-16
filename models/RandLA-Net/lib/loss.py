from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
try:
    from .tools import compute_rotation_matrix_from_ortho6d
except:
    from tools import compute_rotation_matrix_from_ortho6d
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn

loss = nn.CrossEntropyLoss()

#pred_r : batch_size * n * 4 -> batch_size * n * 6
def loss_calculation(end_points):

    pred_seg = end_points["pred_seg"]

    bs, _, num_p = pred_seg.shape

    gt_seg = end_points["gt_seg"]
    gt_seg = gt_seg.view(bs, num_p)

    preds = torch.argmax(pred_seg, dim=1)

    total_pts = preds.shape[0] * preds.shape[1]
    correct_pts = (preds == gt_seg).float().sum()

    acc = correct_pts / total_pts

    semantic_seg_loss = loss(pred_seg, gt_seg)

    end_points["preds"] = preds

    return semantic_seg_loss, acc


class Loss(_Loss):

    def __init__(self):
        super(Loss, self).__init__(True)

    def forward(self, end_points):

        return loss_calculation(end_points)
