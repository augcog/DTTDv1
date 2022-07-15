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
from pointnet2.pointnet2_modules import PointnetSAModuleMSG, PointnetFPModule


psp_models = {
    'resnet18': lambda cfg: PSPNet(cfg=cfg, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda cfg: PSPNet(cfg=cfg, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda cfg: PSPNet(cfg=cfg, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda cfg: PSPNet(cfg=cfg, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda cfg: PSPNet(cfg=cfg, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, cfg, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models[cfg.resnet.lower()](cfg)

    def forward(self, x):
        x = self.model(x)
        return x

class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(
        self, input_channels=6, use_xyz=True, bn=False
    ):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1000,
                radii=[0.0175, 0.025],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
                bn=bn
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=500,
                radii=[0.025, 0.05],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
                bn=bn
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=250,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
                bn=bn
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=125,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
                bn=bn
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512], bn=bn))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        _, N, _ = pointcloud.size()

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]

class PoseNetFeat(nn.Module):
    def __init__(self, cfg, pcld_dim = None):
        super(PoseNetFeat, self).__init__()

        if not pcld_dim:
            pcld_dim = 3 + 3 * cfg.use_normals + 3 * cfg.use_colors

        self.conv1 = torch.nn.Conv1d(pcld_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(cfg.num_points)
        self.num_points = cfg.num_points

    def forward(self, emb, x):

        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))

        #concating them
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))

        #concating them
        pointfeat_2 = torch.cat((x, emb), dim=1)

        #lifting fused 128 + 128 -> 512
        x = F.relu(self.conv5(pointfeat_2))

        #lifting fused 256 + 256 -> 1024
        x = F.relu(self.conv6(x))

        #average pooling on into 1 1024 global feature
        ap_x = self.ap1(x)

        #repeat it so they can staple it onto the back of every pixel/point
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)

        #64 + 64 (level 1), 128 + 128 (level 2), 1024 global feature
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #256 + 512 + 1024

class PoseNet(nn.Module):
    def __init__(self, cfg):
        super(PoseNet, self).__init__()
        self.num_points = cfg.num_points
        self.cnn = ModifiedResnet(cfg)

        if cfg.basic_fusion:
            self.r_out = (pt_utils.Seq(256)
                        .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(cfg.num_objects*6, bn=False, activation=None)
            )

            self.t_out = (pt_utils.Seq(256)
                        .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(cfg.num_objects*3, bn=False, activation=None)
            )

            if cfg.use_confidence:
                self.c_out = (pt_utils.Seq(256)
                            .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                            .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                            .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                            .conv1d(cfg.num_objects*1, bn=False, activation=None)
                )
        else:
            self.r_out = (pt_utils.Seq(1408)
                        .conv1d(640, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(cfg.num_objects*6, bn=False, activation=None)
            )

            self.t_out = (pt_utils.Seq(1408)
                        .conv1d(640, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(cfg.num_objects*3, bn=False, activation=None)
            )

            if cfg.use_confidence:
                self.c_out = (pt_utils.Seq(1408)
                            .conv1d(640, bn=cfg.batch_norm, activation=nn.ReLU())
                            .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                            .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                            .conv1d(cfg.num_objects*1, bn=False, activation=None)
                )

        self.num_obj = cfg.num_objects

        if cfg.pcld_encoder == "pointnet":
            self.df = PoseNetFeat(cfg)
        elif cfg.pcld_encoder == "randlanet":
            self.rndla = RandLANet(cfg=cfg)
            self.df = PoseNetFeat(cfg, pcld_dim=128)
        elif cfg.pcld_encoder == "pointnet2":
            self.pointnet2 = Pointnet2MSG(input_channels=3*cfg.use_normals + 3*cfg.use_colors, bn=cfg.batch_norm)
            self.df = PoseNetFeat(cfg, pcld_dim=128)
        else:
            raise RuntimeError("invalid pcld encoder " + str(cfg.pcld_encoder))

        self.cfg = cfg

        if self.cfg.basic_fusion:
            self.emb_basic = (pt_utils.Seq(32)
                        .conv1d(64, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU()))

    def forward(self, end_points):
        out_img = self.cnn(end_points["img"])
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = end_points["choose"].repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        features = end_points["cloud"]

        if self.cfg.use_normals:
            normals = end_points["normals"]
            features = torch.cat((features, normals), dim=-1)
        
        if self.cfg.use_colors:
            colors = end_points["cloud_colors"]
            features = torch.cat((features, colors), dim=-1)

        if self.cfg.pcld_encoder == "randlanet":
            end_points["RLA_features"] = features.transpose(1, 2)
            end_points = self.rndla(end_points)
            feat_x = end_points["RLA_embeddings"]
        elif self.cfg.pcld_encoder == "pointnet2":
            pcld = features
            feat_x = self.pointnet2(pcld)
        else:
            feat_x = features.transpose(1, 2).contiguous()

        if self.cfg.basic_fusion:
            emb = self.emb_basic(emb)
            ap_x = torch.cat((emb, feat_x), dim=1)
        else:
            ap_x = self.df(emb, feat_x)

        rx = self.r_out(ap_x).view(bs, self.num_obj, 6, self.num_points)
        tx = self.t_out(ap_x).view(bs, self.num_obj, 3, self.num_points)

        if self.cfg.use_confidence:
            cx = torch.sigmoid(self.c_out(ap_x)).view(bs, self.num_obj, 1, self.num_points)

        obj = end_points["obj_idx"].unsqueeze(-1).unsqueeze(-1)
        obj_rx = obj.repeat(1, 1, rx.shape[2], rx.shape[3])
        obj_tx = obj.repeat(1, 1, tx.shape[2], tx.shape[3])
        if self.cfg.use_confidence:
            obj_cx = obj.repeat(1, 1, cx.shape[2], cx.shape[3])

        out_rx = torch.gather(rx, 1, obj_rx)[:,0,:,:]
        out_tx = torch.gather(tx, 1, obj_tx)[:,0,:,:]
        if self.cfg.use_confidence:
            out_cx = torch.gather(cx, 1, obj_cx)[:,0,:,:]

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        if self.cfg.use_confidence:
            out_cx = out_cx.contiguous().transpose(2, 1).contiguous()

        end_points["pred_r"] = out_rx
        end_points["pred_t"] = out_tx
        if self.cfg.use_confidence:
            end_points["pred_c"] = out_cx
        end_points["emb"] = emb.detach()

        return end_points
 
class PoseRefineNetFeat(nn.Module):
    def __init__(self, cfg, pcld_dim=None):
        super(PoseRefineNetFeat, self).__init__()

        if not pcld_dim:
            pcld_dim = 3 + 3 * cfg.use_normals + 3 * cfg.use_colors

        self.conv1 = torch.nn.Conv1d(pcld_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(cfg.num_points)

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, cfg):
        super(PoseRefineNet, self).__init__()
        self.num_points = cfg.num_points

        if cfg.basic_fusion:
            self.r_out = (pt_utils.Seq(256)
                            .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                            .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                            .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                            .conv1d(cfg.num_objects*6, bn=False, activation=None)
                )

            self.t_out = (pt_utils.Seq(256)
                        .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(cfg.num_objects*3, bn=False, activation=None)
            )
        else:
            self.r_out = (pt_utils.Seq(1024)
                        .conv1d(512, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(cfg.num_objects*6, bn=False, activation=None)
            )

            self.t_out = (pt_utils.Seq(1024)
                        .conv1d(512, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(256, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(128, bn=cfg.batch_norm, activation=nn.ReLU())
                        .conv1d(cfg.num_objects*3, bn=False, activation=None)
            )
            

        self.num_obj = cfg.num_objects

        if cfg.pcld_encoder == "pointnet":
            self.feat = PoseRefineNetFeat(cfg)
        elif cfg.pcld_encoder == "randlanet":
            self.rndla = RandLANet(cfg=cfg)
            self.feat = PoseRefineNetFeat(cfg, pcld_dim=128)
        elif cfg.pcld_encoder == "pointnet2":
            self.pointnet2 = Pointnet2MSG(input_channels=3*cfg.use_normals + 3*cfg.use_colors, bn=cfg.batch_norm)
            self.feat = PoseRefineNetFeat(cfg, pcld_dim=128)
        else:
            raise RuntimeError("invalid pcld encoder " + str(cfg.pcld_encoder))

        self.cfg = cfg

        if self.cfg.basic_fusion:
            self.ap1 = torch.nn.AvgPool1d(cfg.num_points)

    def forward(self, end_points, refine_iteration):
        emb = end_points["emb"]
        obj = end_points["obj_idx"]

        bs = obj.size()[0]

        features = end_points["new_points"]

        if self.cfg.use_normals:
            normals = end_points["new_normals"]
            features = torch.cat((features, normals), dim=-1)
        
        if self.cfg.use_colors:
            colors = end_points["cloud_colors"]
            features = torch.cat((features, colors), dim=-1)

        if self.cfg.pcld_encoder == "randlanet":
            end_points["RLA_features"] = features.transpose(1, 2)
            end_points = self.rndla(end_points)
            feat_x = end_points["RLA_embeddings"]
        elif self.cfg.pcld_encoder == "pointnet2":
            pcld = features
            feat_x = self.pointnet2(pcld)
        else:
            feat_x = features.transpose(1, 2)
            
        if self.cfg.basic_fusion:
            ap_x = torch.cat((emb, feat_x), dim=1)
            ap_x = self.ap1(ap_x)
        else:
            ap_x = self.feat(feat_x, emb)

        rx = self.r_out(ap_x).view(bs, self.num_obj, 6, 1)
        tx = self.t_out(ap_x).view(bs, self.num_obj, 3, 1)

        obj = obj.unsqueeze(-1).unsqueeze(-1)
        obj_rx = obj.repeat(1, 1, rx.shape[2], rx.shape[3])
        obj_tx = obj.repeat(1, 1, tx.shape[2], tx.shape[3])

        out_rx = torch.gather(rx, 1, obj_rx)[:,0,:,:]
        out_tx = torch.gather(tx, 1, obj_tx)[:,0,:,:]

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        #print("shapes!", out_rx.shape, out_tx.shape)

        end_points["refiner_pred_r_" + str(refine_iteration)] = out_rx
        end_points["refiner_pred_t_" + str(refine_iteration)] = out_tx

        return end_points
