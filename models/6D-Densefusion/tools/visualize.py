# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import copy
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
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.custom.dataset import PoseDataset as PoseDataset_custom
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine

try:
    from lib.tools import compute_rotation_matrix_from_ortho6d
except:
    from tools import compute_rotation_matrix_from_ortho6d

import open3d as o3d

def visualize_points(model_points, t, rot_mat, label):

    model_points = model_points.cpu().detach().numpy()

    pts = (model_points @ rot_mat.T + t).squeeze()

    pcld = o3d.geometry.PointCloud()
    pts = o3d.utility.Vector3dVector(pts)

    pcld.points = pts
    
    o3d.io.write_point_cloud(label + ".ply", pcld)

def visualize_pointcloud(points, label):

    points = points.cpu().detach().numpy()

    points = points.reshape((-1, 3))

    pcld = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(points)

    pcld.points = points

    o3d.io.write_point_cloud(label + ".ply", pcld)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
    parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
    parser.add_argument('--workers', type=int, default = 2, help='number of data loading workers')
    parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
    parser.add_argument('--model', type=str, default = '',  help='PoseNet model')
    parser.add_argument('--refine_model', type=str, default = '',  help='PoseRefineNet model')
    parser.add_argument('--w', default=0.015, help='regularize confidence')
    parser.add_argument('--w_rate', default=0.3, help='regularize confidence refiner decay')
    parser.add_argument('--num_visualized', type=int, default = 5, help='number of training samples to visualize')
    parser.add_argument('--image_size', type=int, default=25, help="square side length of cropped image")
    opt = parser.parse_args()

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
    elif opt.dataset == 'custom':
        opt.num_objects = 1
        opt.num_points = 500
        opt.outf = 'trained_models/custom'
    else:
        print('Unknown dataset')
        return

    
    if opt.model != '':
        estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
        estimator.cuda()
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.model)))

    if opt.refine_model != '':
        refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
        refiner.cuda()
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.refine_model)))

        #matching train
        opt.w *= opt.w_rate

    print(opt.model)
    print(opt.refine_model)
    if not opt.model and not opt.refine_model:
        raise Exception("this is visualizer code, pls pass in a model lol")

    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_model != '')
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_model != '')
    elif opt.dataset == 'custom':
        test_dataset = PoseDataset_custom('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_model != '', opt.image_size, True)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = test_dataset.get_sym_list()
    opt.num_points_mesh = test_dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the testing set: {0}\nnumber of sample points on mesh: {1}\nsymmetry object list: {2}'.format(len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)

    estimator.eval()

    if opt.refine_model != "":
        refiner.eval()

    dists = []

    for i, data in enumerate(testdataloader, 0):
        points, choose, img, target, model_points, idx = data
        points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                            Variable(choose).cuda(), \
                                                            Variable(img).cuda(), \
                                                            Variable(target).cuda(), \
                                                            Variable(model_points).cuda(), \
                                                            Variable(idx).cuda()
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_model != "")

        bs, num_p, _ = pred_c.shape
        pred_c = pred_c.view(bs, num_p)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_p, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).unsqueeze(0).unsqueeze(0)

        my_rot_mat = compute_rotation_matrix_from_ortho6d(my_r)[0].cpu().data.numpy()

        #print('my rot mat', my_rot_mat)

        my_t = (points.view(bs * num_p, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

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
                
                new_points = torch.bmm((points - T), R).contiguous()
                pred_r, pred_t = refiner(new_points, emb, idx)
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

        my_r = copy.deepcopy(my_rot_mat)

        visualize_points(model_points, my_t, my_r, "{0}_pred".format(i))
        visualize_pointcloud(target, "{0}_target".format(i))
        visualize_pointcloud(points, "{0}_projected_depth".format(i))

        if i >= opt.num_visualized:
            print("finished visualizing!")
            exit()

    print(np.mean(dists))


if __name__ == '__main__':
    main()