#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 22:57:49 2022

@author: endiqq
"""


import math
from collections import deque
import json
from easydict import EasyDict as edict
import os
from functools import partial

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.module import Attention, PreNorm, FeedForward, CrossAttention
import numpy as np
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


from model.classifier import Classifier  # noqa

# print (args.cfg_path)
with open('covid_vit/config/example_PCAM_xq.json') as f:
    cfg = edict(json.load(f))
    # if args.verbose is True:
    #     print(json.dumps(cfg, indent=4))


#%%

# Parallax-Attention Block
class PAB(nn.Module):
    def __init__(self, channels):
        super(PAB, self).__init__()
        # self.head = nn.Sequential(
        #     nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        #     nn.BatchNorm2d(channels),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        #     nn.BatchNorm2d(channels),
        #     nn.LeakyReLU(0.1, inplace=True),
        # )
        self.query = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),
        )
        self.key = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),
        )

    def forward(self, fea_left, fea_right): # left cxr right enh
        '''
        :param x_left:      features from the left image  (B * C * H * W)
        :param x_right:     features from the right image (B * C * H * W)
        :param cost:        input matching cost           (B * H * W * W)
        '''
        b, c, h, w = fea_left.shape
        # fea_left = self.head(x_left)
        # fea_right = self.head(x_right)

        # C_right2left
        Q = self.query(fea_left).permute(0, 2, 3, 1).contiguous()                     # B * H * W * C
        K = self.key(fea_right).permute(0, 2, 1, 3) .contiguous()                     # B * H * C * W
        cost_right2left = torch.matmul(Q, K) / c#(c**0.5)                                      # B * H * W * W
        
        cost_right2left = cost_right2left.softmax(dim=-1)
        # cxr = torch.matmul(fea_left, cost_right2left)
        cxr = torch.matmul(cost_right2left, fea_right.permute(0, 2, 3, 1).contiguous())
        cxr = cxr.permute(0, 3, 1, 2).contiguous()

        # C_left2right
        Q = self.query(fea_right).permute(0, 2, 3, 1).contiguous()                    # B * H * W * C
        K = self.key(fea_left).permute(0, 2, 1, 3).contiguous()                       # B * H * C * W
        cost_left2right = torch.matmul(Q, K) / c#(c**0.5) 
                                     # scale the matching cost
        cost_left2right = cost_left2right.softmax(dim=-1)
        # cxr = torch.matmul(fea_left, cost_right2left)
        enh = torch.matmul(cost_left2right , fea_left.permute(0, 2, 3, 1).contiguous())
        enh = enh.permute(0, 3, 1, 2).contiguous()
        
        return cxr, enh

#%%
class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, args):  #config, 
        super().__init__()
        

        if args.network =='Res50':
            self.cxr_encoder = models.resnet50(pretrained=True)
            self.enh_encoder = models.resnet50(pretrained=True)
            
            # ftrs_vector = model_cxr.fc.in_features
            
            # self.cxr_encoder = nn.Sequential(*list(model_cxr.children()))[:-2]
            # self.enh_encoder = nn.Sequential(*list(model_enh.children()))[:-2]
            
        elif args.network == 'dense121':
            model_cxr = models.densenet121(pretrained=True)
            model_enh = models.densenet121(pretrained=True)
            
            ftrs_vector = model_cxr.classifier.in_features
            # self.conv1_enh = nn.Conv(1024, config.n_embd, kernel_size=1)
            self.cxr_encoder = model_cxr.features
            self.enh_encoder = model_enh.features

        elif args.network == 'res34_dense121':
            
            self.cxr_encoder = models.resnet34(pretrained=True)
            ftrs_vector_cxr = self.cxr_encoder.fc.in_features
            
            # model_cxr = models.resnet50(pretrained=True)
            model_enh = models.densenet121(pretrained=True)
            ftrs_vector_enh = model_enh.classifier.in_features
            # self.conv1_enh = nn.Conv(1024, config.n_embd, kernel_size=1)
            self.enh_encoder = model_enh.features
            
        elif args.network == 'chexpert':
            
            ftrs_vector = 1024
            model_path = 'CheXpert_logdir'
            
            model_cxr = Classifier(cfg)
            ckpt_path = os.path.join(model_path, 'best3_CXR.ckpt')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model_cxr.load_state_dict(ckpt['state_dict'])
            self.cxr_encoder = model_cxr.backbone
            
            model_enh = Classifier(cfg)
            ckpt_path = os.path.join(model_path, 'best2_Enh.ckpt')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model_enh.load_state_dict(ckpt['state_dict'])
            self.enh_encoder = model_enh.backbone
        
        self.pa1 = PAB(256)
        self.pa2 = PAB(512)
        self.pa3 = PAB(1024)

        
    def forward(self, cxr_image, enh_image):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''

        # 1
        image_features = self.cxr_encoder.conv1(cxr_image)
        image_features = self.cxr_encoder.bn1(image_features)
        image_features = self.cxr_encoder.relu(image_features)
        image_features = self.cxr_encoder.maxpool(image_features)
        image_features = self.cxr_encoder.layer1(image_features)
        
        lidar_features = self.enh_encoder.conv1(enh_image)
        lidar_features = self.enh_encoder.bn1(lidar_features)
        lidar_features = self.enh_encoder.relu(lidar_features)
        lidar_features = self.enh_encoder.maxpool(lidar_features)       
        lidar_features = self.enh_encoder.layer1(lidar_features)

        
        image_features_layer1, lidar_features_layer1 = self.pa1(image_features, lidar_features)
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1
        
        # 2
        image_features = self.cxr_encoder.layer2(image_features)# output [2, 128, 28, 28]
        lidar_features = self.enh_encoder.layer2(lidar_features) #output [2, 512, 28, 28]      
        image_features_layer2, lidar_features_layer2 = self.pa2(image_features, lidar_features)
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        # 3
        image_features = self.cxr_encoder.layer3(image_features) #output [2, 256, 14, 14]
        lidar_features = self.enh_encoder.layer3(lidar_features) #output [2, 1024, 14, 14]        
        image_features_layer3, lidar_features_layer3 = self.pa3(image_features, lidar_features)
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3

        # 4
        image_features = self.cxr_encoder.layer4(image_features) #output [2, 512, 7, 7]
        lidar_features = self.enh_encoder.layer4(lidar_features) #output [2, 1024, 7, 7]

        return image_features, lidar_features
        # return fused_features

#%%
class ClassificationHead(nn.Module):
    def __init__(self, num_classes, large_dim):
        super(ClassificationHead, self).__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(large_dim, eps=1e-6)
        # self.embed_dim = large_dim
        
        for index, num_class in enumerate(num_classes):
            setattr(self, "fc_" + str(index), nn.Linear(large_dim, num_class))

    def forward(self, x):
        feat_map = self.norm(x)
        logits = list()
        for index, num_class in enumerate(self.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            logit = classifier(feat_map)
            logits.append(logit)

        return logits 
    

class ms_transfuser_mid(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, args, num_classes, embed_dim): #config,
        super().__init__()
        
        self.encoder = Encoder(args)#.to(self.device)
        
        self.fusion_conv = nn.Sequential(nn.Conv2d(2*embed_dim, embed_dim, kernel_size=1),
                             nn.BatchNorm2d(2048),
                             nn.ReLU(inplace=True),)
        
        self.output = ClassificationHead(num_classes, embed_dim)
        # self.mlp_head_enh = ClassificationHead(num_classes, large_dim)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, img_cxr, img_enh):
                
        xs, xl = self.encoder(img_cxr, img_enh)

        x3 = torch.cat((xs, xl), dim=1)
        x3 = self.fusion_conv(x3)
        # print (x3.shape) 
        x3 = F.adaptive_avg_pool2d(x3, (1, 1)).squeeze(2).squeeze(2)        
        # print (x3.shape)
        logits = self.output(x3)

        return logits