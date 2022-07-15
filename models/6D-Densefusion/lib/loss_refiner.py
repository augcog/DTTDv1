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
from knn_cuda import KNN

from lib.loss_helpers import FRONT_LOSS_COEFF

def loss_calculation(end_points, refine_iteration, num_point_mesh, sym_list, use_normals):

    pred_r = end_points["refiner_pred_r_" + str(refine_iteration)]
    pred_t = end_points["refiner_pred_t_" + str(refine_iteration)]
    target = end_points["new_target"]
    target_front = end_points["new_target_front"]
    model_points = end_points["model_points"]
    front = end_points["front"]
    points = end_points["new_points"]
    idx = end_points["obj_idx"]

    if use_normals:
        normals = end_points["new_normals"]

    knn = KNN(k=1, transpose_mode=True)
    bs, num_p, _ = pred_r.size()
    num_input_points = len(points[0])

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))

    base = compute_rotation_matrix_from_ortho6d(pred_r)
    base = base.view(bs*num_p, 3, 3)

    
    # base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
    #                   (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
    #                   (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
    #                   (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    front = front.view(bs, 1, 1, 3).repeat(1, num_p, 1, 1).view(bs * num_p, 1, 3)

    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs, num_p, num_point_mesh, 3)
    target_front = target_front.view(bs, 1, 1, 3).repeat(1, num_p, 1, 1)

    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t


    #print("shapes before bmm?", model_points.shape, base.shape, points.shape, pred_t.shape)

    pred = torch.add(torch.bmm(model_points, base), pred_t)
    pred_front = torch.add(torch.bmm(front, base), pred_t)

    pred = pred.view(bs, num_p, num_point_mesh, 3)

    #print("loss shapes now before dist calc", pred.shape, target.shape)

    for i in range(len(idx)):
        if idx[i].item() in sym_list:

            my_target = target[i,0,:,:].contiguous().view(1, -1, 3)
            my_pred = pred[i].contiguous().view(1, -1, 3)

            dists, inds = knn(my_target, my_pred)

            inds = inds.repeat(1, 1, 3)

            my_target = torch.gather(my_target, 1, inds)
            my_target = my_target.view(num_p, num_point_mesh, 3).contiguous()
            target[i] = my_target

    target = target.detach()

    #print("shapes before diff", pred.shape, target.shape)
    
    dis = torch.mean(torch.norm((pred - target), dim=3), dim=2)
    dis_front = torch.mean(torch.norm((pred_front - target_front), dim=3), dim=2)
    
    loss = torch.sum(torch.mean(dis + FRONT_LOSS_COEFF * dis_front, dim=1))

    #print("dis shape", dis.shape)

    dis = dis.view(bs, num_p)

    #print("gotta do this stuff now", ori_t.shape, ori_base.shape, points.shape)

    t = ori_t[:,0]

    points = points.contiguous().view(bs, num_input_points, 3)

    ori_base = ori_base.view(bs, 3, 3).contiguous()
    ori_t = t.repeat(1, num_input_points, 1).contiguous().view(bs, num_input_points, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    if use_normals:
        normals = normals.view(bs, num_input_points, 3)
        new_normals = torch.bmm(normals, ori_base).contiguous()

    new_target = ori_target.view(bs, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(1, num_point_mesh, 1).contiguous().view(bs, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    new_target_front = target_front.view(bs, 1, 3).contiguous()
    ori_t = t.view(bs, 1, 3)
    new_target_front = torch.bmm((new_target_front - ori_t), ori_base).contiguous()
 
    dis = torch.mean(dis)

    end_points["new_points"] = new_points.detach()
    if use_normals:
        end_points["new_normals"] = new_normals.detach()
    end_points["new_target"] = new_target.detach()
    end_points["new_target_front"] = new_target_front.detach()

    # print('------------> ', dis.item(), idx[0].item())
    del knn
    return loss, dis, end_points


class Loss_refine(_Loss):

    def __init__(self, num_points_mesh, sym_list, use_normals):
        super(Loss_refine, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list
        self.use_normals = use_normals

    def forward(self, end_points, refine_iteration):
        return loss_calculation(end_points, refine_iteration, self.num_pt_mesh, self.sym_list, self.use_normals)
