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
from cfg.config import YCBConfig as Config, write_config

from collections import defaultdict
from knn_cuda import KNN

from lib.randla_utils import randla_processing

def cal_auc(add_dis):
        max_dis = 0.1
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf;
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
        aps = VOCap(D, acc)
        return aps * 100

def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

try:
    from lib.tools import compute_rotation_matrix_from_ortho6d
except:
    from tools import compute_rotation_matrix_from_ortho6d

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--use_posecnn_rois', action="store_true", default=False, help="use the posecnn roi's")
opt = parser.parse_args()

cfg = Config()
cfg.refine_start = opt.refine_model != ''

batch_size = 1
workers = 1
cfg.posecnn_results = "YCB_Video_toolbox/results_PoseCNN_RSS2018"

if opt.use_posecnn_rois:
    from datasets.ycb.dataset import PoseDatasetPoseCNNResults as PoseDataset
else:
    from datasets.ycb.dataset import PoseDatasetAllObjects as PoseDataset

def get_pointcloud(model_points, t, rot_mat):

    model_points = model_points.cpu().detach().numpy()

    pts = (model_points @ rot_mat.T + t).squeeze()

    return pts
    
def main():
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

    adds = defaultdict(list)
    add = defaultdict(list)

    knn = KNN(k=1, transpose_mode=True)

    test_dist = 0
    test_count = 0

    sym_list = test_dataset.get_sym_list()

    with torch.no_grad():
        for now, data_objs in enumerate(test_dataloader):
            for obj_idx, end_points in enumerate(data_objs):
                torch.cuda.empty_cache()

                del end_points["intr"]

                end_points_cuda = {}
                for k, v in end_points.items():
                    end_points_cuda[k] = Variable(v).cuda()

                end_points = end_points_cuda

                if cfg.pcld_encoder == "randlanet":
                    end_points = randla_processing(end_points, cfg)
                                                                        
                end_points = estimator(end_points)

                pred_r = end_points["pred_r"]
                pred_t = end_points["pred_t"]
                pred_c = end_points["pred_c"]
                points = end_points["cloud"] + end_points["cloud_mean"]
                model_points = end_points["model_points"]
                target = end_points["target"]
                idx = end_points["obj_idx"]

                bs, num_p, _ = pred_t.shape

                if cfg.use_normals:
                    normals = end_points["normals"]

                if cfg.use_confidence:
                    pred_c = end_points["pred_c"]
                    pred_c = pred_c.view(bs, num_p)
                    how_max, which_max = torch.max(pred_c, 1)
                    pred_t = pred_t.view(bs * num_p, 1, 3)

                    my_r = pred_r[0][which_max[0]].view(-1).unsqueeze(0).unsqueeze(0)

                    my_rot_mat = compute_rotation_matrix_from_ortho6d(my_r)[0].cpu().data.numpy()

                    points = points.contiguous().view(bs*num_p, 1, 3)

                    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                else:

                    my_r = torch.mean(pred_r, dim=1, keepdim=True)
                    pred_t = points + pred_t
                    my_t = torch.mean(pred_t, dim=1).view(-1).cpu().data.numpy()
                    my_rot_mat = compute_rotation_matrix_from_ortho6d(my_r)[0].cpu().data.numpy()

                #print('my rot mat', my_rot_mat)

                points = points.contiguous().view(bs*num_p, 1, 3)

                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

                if opt.refine_model != "":
                    for ite in range(0, opt.iteration):
                        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_p, 1).contiguous().view(1, num_p, 3)
                        rot_mat = my_rot_mat

                        #print('iter!', ite, my_rot_mat)

                        my_mat = np.zeros((4,4))
                        my_mat[0:3,0:3] = rot_mat
                        my_mat[3, 3] = 1

                        #print(my_mat, my_mat.shape)

                        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                        my_mat[0:3, 3] = my_t

                        points = points.view(bs, num_p, 3)
                        
                        new_points = torch.bmm((points - T), R).contiguous()
                        end_points["new_points"] = new_points.detach()

                        if opt.use_normals:
                            normals = normals.view(bs, num_p, 3)
                            new_normals = torch.bmm(normals, R).contiguous()
                            end_points["new_normals"] = new_normals.detach()

                        end_points = refiner(end_points, ite)

                        pred_r = end_points["refiner_pred_r_" + str(ite)]
                        pred_t = end_points["refiner_pred_t_" + str(ite)]
            
                        pred_r = pred_r.view(1, 1, -1)
                        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                        my_r_2 = pred_r.view(-1).unsqueeze(0).unsqueeze(0)
                        my_t_2 = pred_t.view(-1).cpu().data.numpy()
                        rot_mat_2 = compute_rotation_matrix_from_ortho6d(my_r_2)[0].cpu().data.numpy()

                        my_mat_2 = np.zeros((4, 4))
                        my_mat_2[0:3,0:3] = rot_mat_2
                        my_mat_2[3,3] = 1
                        my_mat_2[0:3, 3] = my_t_2

                        my_mat_final = np.dot(my_mat, my_mat_2)
                        my_r_final = copy.deepcopy(my_mat_final)
                        my_rot_mat[0:3,0:3] = my_r_final[0:3,0:3]
                        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                        #my_pred = np.append(my_r_final, my_t_final)
                        my_t = my_t_final
                # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

                my_r = copy.deepcopy(my_rot_mat)
                pred = get_pointcloud(model_points, my_t, my_r)
                pred = torch.unsqueeze(torch.from_numpy(pred.astype(np.float32)), 0).cuda()

                add_dist = torch.mean(torch.norm(target - pred, dim=2)).detach().cpu().item()

                adds_dists, inds = knn(target, pred)
                adds_dist = torch.mean(adds_dists).detach().cpu().item()
                idx = idx.detach().cpu().item()

                if idx in sym_list:
                    test_dist += adds_dist
                else:
                    test_dist += add_dist
                test_count += 1

                print("frame, idx, adds, add", now, idx, adds_dist, add_dist)

                adds[idx].append(adds_dist)
                add[idx].append(add_dist)

                for d in end_points:
                    del d

    print("DIST!", (test_dist / test_count))

    adds_aucs = {}
    add_aucs = {}

    for idx, dists in adds.items():
        adds_aucs[idx] = cal_auc(dists)

    for idx, dists in add.items():
        add_aucs[idx] = cal_auc(dists)

    print("ADDS AUCs")
    print(adds_aucs)

    print("ADD AUCs")
    print(add_aucs)

if __name__ == "__main__":
    main()
