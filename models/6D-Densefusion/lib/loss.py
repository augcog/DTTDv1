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

from scipy.spatial.transform import Rotation as R

from lib.loss_helpers import FRONT_LOSS_COEFF, average_quaternion


#pred_r : batch_size * n * 4 -> batch_size * n * 6
def loss_calculation(end_points, w, refine, num_point_mesh, sym_list, use_normals, use_confidence):

    pred_r = end_points["pred_r"]
    pred_t = end_points["pred_t"]
    if use_confidence:
        pred_c = end_points["pred_c"]
    target = end_points["target"]
    target_front = end_points["target_front"]
    model_points = end_points["model_points"]
    front = end_points["front"]
    points = end_points["cloud"] + end_points["cloud_mean"]
    idx = end_points["obj_idx"]
    
    if use_normals:
        normals = end_points["normals"]

    #print("shapes loss regular", pred_r.shape, pred_t.shape, pred_c.shape, target.shape, model_points.shape, points.shape)

    knn = KNN(k=1, transpose_mode=True)

    bs, num_p, _ = pred_t.size()

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
    pred_t = pred_t.contiguous().view(bs*num_p, 1, 3)
    ori_t = pred_t

    points = points.contiguous().view(bs*num_p, 1, 3)
    
    if use_confidence:
        pred_c = pred_c.contiguous().view(bs, num_p)

    pred = torch.add(torch.bmm(model_points, base), points + pred_t)
    pred_front = torch.add(torch.bmm(front, base), points + pred_t)

    pred = pred.view(bs, num_p, num_point_mesh, 3)
    pred_front = pred_front.view(bs, num_p, 1, 3)

    #print("loss shapes now before possible knn", pred.shape, target.shape)

    #knn will happen in refiner loss
    if not refine:
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

    dis = torch.mean(torch.norm((pred - target), dim=3), dim=2)
    front_dis = torch.mean(torch.norm((pred_front - target_front), dim=3), dim=2)

    if use_confidence:
        loss = torch.sum(torch.mean(((dis + FRONT_LOSS_COEFF * front_dis) * pred_c - w * torch.log(pred_c)), dim=1))
        pred_c = pred_c.view(bs, num_p)
        how_max, which_max = torch.max(pred_c, 1)

        ori_t = ori_t.view(bs, num_p, 1, 3)
        points = points.view(bs, num_p, 1, 3)

        ori_which_max = which_max

        which_max = which_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
        #print("gotta do this stuff now", ori_t.shape, points.shape, which_max.shape)

        t = torch.gather(ori_t, 1, which_max) + torch.gather(points, 1, which_max)#ori_t[:,which_max] + points[:,which_max]

        ori_base = ori_base.view(bs, num_p, 3, 3)

        #print("more this stuff now", ori_base.shape)

        which_max = ori_which_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)

        ori_base = torch.gather(ori_base, 1, which_max).view(bs, 3, 3).contiguous()
        ori_t = t.repeat(1, num_p, 1, 1).contiguous()
    else:
        loss = torch.sum(torch.mean(((dis + FRONT_LOSS_COEFF * front_dis)), dim=1))

        ori_t = ori_t.view(bs, num_p, 1, 3)
        points = points.view(bs, num_p, 1, 3)
        ori_base = ori_base.view(bs, num_p, 3, 3)

        #use quaternion averaging now for average rotation
        ori_base = ori_base.cpu().detach().numpy()
        ori_base_avgs = np.zeros((bs, 3, 3)).astype(np.float32)
        for i in range(bs):
            quat = R.from_matrix(ori_base[i]).as_quat()
            avg_quat = average_quaternion(quat)
            avg_rot_mat = R.from_quat(avg_quat).as_matrix()
            ori_base_avgs[i] = avg_rot_mat
        ori_base = torch.from_numpy(ori_base_avgs).cuda().contiguous()
        ori_t = torch.mean(ori_t + points, dim=1, keepdim=True)

        t = ori_t.contiguous()

        ori_t = ori_t.repeat(1, num_p, 1, 1).contiguous()

    #ori_t = translation pred
    #ori_base = rotation pred
    
    dis = dis.view(bs, num_p)

    ori_t = ori_t.view(bs, num_p, 3)
    points = points.view(bs, num_p, 3)

    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    if use_normals:
        normals = normals.contiguous().view(bs, num_p, 3)
        new_normals = torch.bmm(normals, ori_base).contiguous()

    new_target = ori_target[:,0].view(bs, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(1, num_point_mesh, 1, 1).contiguous().view(bs, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    new_target_front = target_front[:,0].view(bs, 1, 3).contiguous()
    ori_t = t.view(bs, 1, 3)
    new_target_front = torch.bmm((new_target_front - ori_t), ori_base).contiguous()

    # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())

    #print("outputting this thingy", dis.shape, ori_which_max.shape)

    if use_confidence:
        which_max = ori_which_max.unsqueeze(-1)
        dis = torch.gather(dis, 1, which_max)
        dis = torch.mean(dis)
    else:
        dis = torch.mean(dis)

    end_points["new_points"] = new_points.detach()
    if use_normals:
        end_points["new_normals"] = new_normals.detach()
    end_points["new_target"] = new_target.detach()
    end_points["new_target_front"] = new_target_front.detach()

    del knn
    return loss, dis, end_points


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list, use_normals, use_confidence):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list
        self.use_normals = use_normals
        self.use_confidence = use_confidence

    def forward(self, end_points, w, refine):

        return loss_calculation(end_points, w, refine, self.num_pt_mesh, self.sym_list, self.use_normals, self.use_confidence)
