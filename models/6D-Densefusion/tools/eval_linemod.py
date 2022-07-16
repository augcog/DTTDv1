import _init_paths
import argparse
import os
import random
import numpy as np
import yaml
import copy
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
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from knn_cuda import KNN
from PIL import Image
import cv2

try:
    from lib.tools import compute_rotation_matrix_from_ortho6d
except:
    from tools import compute_rotation_matrix_from_ortho6d

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--output', type=str, default='visualization', help='output for point vis')
opt = parser.parse_args()

if not os.path.isdir(opt.output):
    os.mkdir(opt.output)


num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
iteration = 4
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'
knn = KNN(k=1, transpose_mode=True)


estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator = nn.DataParallel(estimator)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner = nn.DataParallel(refiner)
refiner.cuda()

print("model?", opt.model)

estimator_dict = torch.load(opt.model)
estimator_dict_parallel = {}
for k, v in estimator_dict.items():
    estimator_dict_parallel["module." + k] = v

estimator.load_state_dict(estimator_dict_parallel)

refiner_dict = torch.load(opt.refine_model)
refiner_dict_parallel = {}
for k, v in refiner_dict.items():
    refiner_dict_parallel["module." + k] = v

refiner.load_state_dict(refiner_dict_parallel)

estimator.eval()
refiner.eval()

testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)

testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(diameter)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

def project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy):
    proj_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
    projected_pts = pts @ proj_mat.T
    projected_pts /= np.expand_dims(projected_pts[:,2], -1)
    projected_pts = projected_pts[:,:2]
    return projected_pts

for i, data in enumerate(testdataloader, 0):

    color_img_file = testdataset.list_rgb[i]
    color_img = cv2.imread(color_img_file)

    points, choose, img, target, model_points, idx = data
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                     Variable(choose).cuda(), \
                                                     Variable(img).cuda(), \
                                                     Variable(target).cuda(), \
                                                     Variable(model_points).cuda(), \
                                                     Variable(idx).cuda()

    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).unsqueeze(0).unsqueeze(0)

    my_rot_mat = compute_rotation_matrix_from_ortho6d(my_r)[0].cpu().data.numpy()

    #print('my rot mat', my_rot_mat)

    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    #my_pred = np.append(my_r, my_t)

    for ite in range(0, iteration):
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
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

    # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

    model_points = model_points[0].cpu().detach().numpy()
    my_r = copy.deepcopy(my_rot_mat)
    pred = np.dot(model_points, my_r.T) + my_t
    target = target[0].cpu().detach().numpy()

    projected_pred = project_points(pred, testdataset.cam_fx, testdataset.cam_fy, testdataset.cam_cx, testdataset.cam_cy)

    for (x, y) in projected_pred:
        color_img = cv2.circle(color_img, (int(x), int(y)), radius=1, color=(0,255,0), thickness=-1)

    output_filename = '{0}/{1}.png'.format(opt.output, i)
    cv2.imwrite(output_filename, color_img)



    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).cuda()
        target = torch.from_numpy(target.astype(np.float32)).unsqueeze(0).cuda()
        
        dists, inds = knn(target, pred)
        target = torch.index_select(target, 1, inds.view(-1))
    
        diffs = torch.norm((pred - target), dim = 2).cpu().numpy()
        dis = np.mean(diffs)

    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    if dis < 0.02:
        success_count[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1
    # print('No.{0} examed'.format(idx[0].item()))

for i in range(num_objects):
    if num_count[i] != 0:
        print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
        fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.close()
