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

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from datasets.ak_ip.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
from lib.randla_utils import randla_processing
from cfg.config import AKIPConfig as Config, write_config
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ExponentialLR

import faulthandler
faulthandler.enable()

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

cfg = Config()

def main():
    cfg.manualSeed = random.randint(1, 10000)
    random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)

    estimator = PoseNet(cfg = cfg)
    estimator.cuda()
    refiner = PoseRefineNet(cfg = cfg)
    refiner.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(cfg.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        print("performing refiner training!")
        refiner.load_state_dict(torch.load('{0}/{1}'.format(cfg.outf, opt.resume_refinenet)))
        cfg.refine_start = True
        cfg.decay_start = True
        cfg.w *= cfg.w_rate
        optimizer = optim.Adam(refiner.parameters(), lr=cfg.lr)
    else:
        cfg.refine_start = False
        cfg.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=cfg.lr)

    dataset = PoseDataset('train', cfg = cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate_fn, shuffle=True, num_workers=cfg.workers)
    test_dataset = PoseDataset('test', cfg = cfg)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate_fn, shuffle=False, num_workers=cfg.workers)
    
    cfg.sym_list = dataset.get_sym_list()
    cfg.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), cfg.num_points_mesh, cfg.sym_list))

    criterion = Loss(cfg.num_points_mesh, cfg.sym_list, cfg.use_normals, cfg.use_confidence)
    criterion_refine = Loss_refine(cfg.num_points_mesh, cfg.sym_list, cfg.use_normals)

    if cfg.lr_scheduler == "cyclic":
        clr_div = 6
        lr_scheduler = CyclicLR(
            optimizer, base_lr=1e-5, max_lr=3e-4,
            cycle_momentum=False,
            step_size_up=cfg.nepoch * (len(dataset) / cfg.batch_size) // clr_div,
            step_size_down=cfg.nepoch * (len(dataset) / cfg.batch_size) // clr_div,
            mode='triangular'
        )
    elif cfg.lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.nepoch * (len(dataset) / cfg.batch_size))
    elif cfg.lr_scheduler == "exponential":
        lr_scheduler = ExponentialLR(optimizer, 0.9)
    else:
        lr_scheduler = None

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(cfg.log_dir):
            if ".gitignore" in log:
                continue
            os.remove(os.path.join(cfg.log_dir, log))
    st_time = time.time()

    for epoch in range(opt.start_epoch, cfg.nepoch):

        faulthandler.dump_traceback_later(90 * 60) #90 minutes (catch deadlock)

        write_config(cfg, os.path.join(cfg.log_dir, "config_current.yaml"))
        logger = setup_logger('epoch%d' % epoch, os.path.join(cfg.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        if cfg.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(cfg.repeat_epoch):
            trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="training")
            for batch_id, end_points in trange:
                start_time = time.time()

                if len(end_points) == 0:
                    continue

                end_points_cuda = {}
                for k, v in end_points.items():
                    end_points_cuda[k] = Variable(v).cuda()

                end_points = end_points_cuda

                if cfg.pcld_encoder == "randlanet":
                    end_points = randla_processing(end_points, cfg)

                end_points = estimator(end_points)

                loss, dis, end_points = criterion(end_points, cfg.w, cfg.refine_start)

                if cfg.refine_start:
                    for ite in range(0, cfg.iteration):
                        end_points = refiner(end_points, ite)

                        loss, dis, end_points = criterion_refine(end_points, ite)
                        loss.backward()
                else:
                    loss.backward()

                dis = dis.item()
                train_dis_avg += dis
                train_count += 1
                trange.set_postfix(dis=(train_dis_avg / train_count))

                optimizer.step()
                optimizer.zero_grad()

                if batch_id != 0 and batch_id % (len(dataloader) // 40) == 0:
                    logger.info('Epoch {} | Batch {} | dis:{}'.format(epoch, batch_id, dis))
                    if cfg.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(cfg.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(cfg.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        if lr_scheduler:
            lr_scheduler.step()

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(cfg.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        trange = tqdm(enumerate(testdataloader), total=len(testdataloader), desc="testing")

        with torch.no_grad():
            for batch_id, end_points in trange:

                end_points_cuda = {}
                for k, v in end_points.items():
                    end_points_cuda[k] = Variable(v).cuda()

                end_points = end_points_cuda

                if cfg.pcld_encoder == "randlanet":
                    end_points = randla_processing(end_points, cfg)
                
                end_points = estimator(end_points)

                _, dis, end_points = criterion(end_points, cfg.w, cfg.refine_start)

                if cfg.refine_start:
                    for ite in range(0, cfg.iteration):
                        end_points = refiner(end_points, ite)

                        _, dis, end_points = criterion_refine(end_points, ite)
                dis = dis.item()
                test_dis += dis
                trange.set_postfix(dis=dis)
                test_count += 1

        test_dis = test_dis / test_count
        logger.info('Epoch {} TEST FINISH Avg dis: {}'.format(epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            if cfg.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(cfg.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(cfg.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < cfg.decay_margin and not cfg.decay_start:
            cfg.decay_start = True
            cfg.w *= cfg.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=cfg.lr)

        if (epoch >= cfg.refine_epoch or best_test < cfg.refine_margin) and not cfg.refine_start:
            cfg.refine_start = True
            optimizer = optim.Adam(refiner.parameters(), lr=cfg.lr)

            if cfg.dataset == 'ycb':
                dataset = PoseDataset('train', cfg = cfg)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
            if cfg.dataset == 'ycb':
                test_dataset = PoseDataset('test', cfg = cfg)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
            
            cfg.sym_list = dataset.get_sym_list()
            cfg.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), cfg.num_points_mesh, cfg.sym_list))

            criterion = Loss(cfg.num_points_mesh, cfg.sym_list, cfg.use_normals, cfg.use_confidence)
            criterion_refine = Loss_refine(cfg.num_points_mesh, cfg.sym_list, cfg.use_normals)

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()
