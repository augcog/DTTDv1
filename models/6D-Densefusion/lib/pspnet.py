import torch
from torch import nn
from torch.nn import functional as F
import lib.extractors as extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(PSPUpsample, self).__init__()
        conv_layers = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)]

        if bn:
            conv_layers.append(nn.BatchNorm2d(out_channels))
            
        conv_layers.append(nn.PReLU())

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv(x)


class PSPNet(nn.Module):
    def __init__(self, cfg, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18'):
        super(PSPNet, self).__init__()
        self.feats = getattr(extractors, backend)(cfg)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256, bn=cfg.batch_norm)
        self.up_2 = PSPUpsample(256, 64, bn=cfg.batch_norm)
        self.up_3 = PSPUpsample(64, 64, bn=cfg.batch_norm)

        self.drop_2 = nn.Dropout2d(p=0.15)

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1)
        )

    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        
        return self.final(p)

