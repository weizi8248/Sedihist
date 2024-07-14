##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn

from .nn import Encoding, View, Normalize
from backbone import resnet50s, resnet101s, resnet152s

__all__ = ['DeepTen']

class DeepTen(nn.Module):
    def __init__(self, nclass=9, backbone='resnet50'):
        super(DeepTen, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if self.backbone == 'resnet50':
            self.pretrained = resnet50s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152s(pretrained=True, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))
        n_codes = 32
        self.head = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128,K=n_codes),
            View(-1, 128*n_codes),
            Normalize(),
            nn.Linear(128*n_codes, nclass),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        return self.head(x)

