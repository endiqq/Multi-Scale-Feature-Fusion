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
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    
class MultiScaleTransformerEncoder(nn.Module):
    # img_size = 512; small_dim=large_dim=256; img_size = 224; small_dim=large_dim=49
    def __init__(self, small_dim = 256, small_depth = 2, small_heads =8, small_dim_head = 32, small_mlp_dim = 384,
                 large_dim = 256,  large_depth = 2, large_heads = 8, large_dim_head = 64, large_mlp_dim = 768,
                 cross_attn_depth = 1, cross_attn_heads = 8, dropout = 0.):
        super().__init__()
        
        self.transformer_enc_small = Transformer(small_dim, small_depth, small_heads, small_dim_head, small_mlp_dim)
        self.transformer_enc_large = Transformer(large_dim, large_depth, large_heads, large_dim_head, large_mlp_dim)        

        self.cross_attn_layers = nn.ModuleList([]) 
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, num_heads = cross_attn_heads, attn_drop = dropout)),
                # nn.LayerNorm(large_dim, eps=1e-6),
                
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, num_heads = cross_attn_heads, attn_drop = dropout)),
                # nn.LayerNorm(small_dim, eps=1e-6),
                
            ]))

    def forward(self, xs, xl):
        # print (len(xs), xs[0].shape)
        xs = self.transformer_enc_small(xs)
        xl = self.transformer_enc_large(xl)

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch

            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        return xs, xl


# Parallax-Attention Block
class PAB(nn.Module):
    def __init__(self, channels, config):
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
        
        # self.value = nn.Sequential(
        #     nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
        #     nn.BatchNorm2d(channels),
        # )

        # self.pos_emb= nn.Parameter(torch.zeros(1, config.vert_anchors * config.horz_anchors, channels))
        # self.pos_emb_cxr = nn.Parameter(torch.zeros(1, config.vert_anchors * config.horz_anchors, channels))
        # self.pos_emb_enh = nn.Parameter(torch.zeros(1, config.vert_anchors * config.horz_anchors, channels))

    def forward(self, fea_left, fea_right): # left cxr right enh
        '''
        :param x_left:      features from the left image  (B * C * H * W)
        :param x_right:     features from the right image (B * C * H * W)
        :param cost:        input matching cost           (B * H * W * W)
        '''
        b, c, h, w = fea_left.shape
        # print (fea_left.shape)
        # fea_left = self.pos_emb+fea_left
        # fea_right = self.pos_emb+fea_right
        # fea_left = self.pos_emb_cxr+fea_left
        # fea_right = self.pos_emb_enh+fea_right

        # C_right2left
        # Q = self.query(fea_left).permute(0, 2, 3, 1).contiguous()                     # B * H * W * C
        # K = self.key(fea_right).permute(0, 2, 1, 3) .contiguous()                     # B * H * C * W
        # cost_right2left = torch.matmul(Q, K) / c   #cxr                                   # B * H * W * W
        
        Q = self.query(fea_left).view(b, -1, c)#.contiguous()                    # B * H * W * C
        # print (Q.shape)
        K = self.key(fea_right).view(b, -1, c) #.contiguous()                     # B * H * C * W
        # print (K.shape)
        cost_right2left = torch.matmul(Q, K.transpose(1,2)) / c   #cxr                                   # B * H * W * W
        
        cost_right2left = cost_right2left.softmax(dim=-1)
        # print (cost_right2left.shape)
        # cxr = torch.matmul(fea_left, cost_right2left)
        # cxr = torch.matmul(cost_right2left, self.value(fea_right))
        cxr = torch.matmul(cost_right2left, fea_right.view(b, -1, c))
        cxr = cxr.transpose(1, 2).contiguous().view(b, c, h, w) # re-assemble all head outputs side by side

        # C_left2right
        Q = self.query(fea_right).view(b, -1, c)#.contiguous()                    # B * H * W * C
        K = self.key(fea_left).view(b, -1, c)#.contiguous()                       # B * H * C * W
        cost_left2right = torch.matmul(Q, K.transpose(1,2)) / c #enh
        #                              # scale the matching cost
        cost_left2right = cost_left2right.softmax(dim=-1)
        # cxr = torch.matmul(fea_left, cost_right2left)
        # enh = torch.matmul(cost_left2right, self.value(fea_left))
        enh = torch.matmul(cost_left2right, fea_left.view(b, -1, c))
        enh = enh.transpose(1, 2).contiguous().view(b, c, h, w)
        
        return cxr, enh
        # return cost_right2left, cost_left2right

#%%
class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, args, config):  #config, 
        super().__init__()
        
        self.config = config
        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        
        if args.network =='Res50':
            model_cxr = models.resnet50(pretrained=True)
            model_enh = models.resnet50(pretrained=True)
            
            ftrs_vector = model_cxr.fc.in_features
            
            self.cxr_encoder = nn.Sequential(*list(model_cxr.children()))[:-2]
            self.enh_encoder = nn.Sequential(*list(model_enh.children()))[:-2]
            
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
        
        self.pa1 = PAB(128, config)
        self.pa2 = PAB(256, config)
        self.pa3 = PAB(512, config)

        
    def forward(self, cxr_image, enh_image):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''

        # 1
        image_features = self.cxr_encoder.features.conv0(cxr_image)
        image_features = self.cxr_encoder.features.norm0(image_features)
        image_features = self.cxr_encoder.features.relu0(image_features)
        image_features = self.cxr_encoder.features.pool0(image_features)
        image_features = self.cxr_encoder.features.denseblock1(image_features) 
        image_features = self.cxr_encoder.features.transition1(image_features)        
        
        lidar_features = self.enh_encoder.features.conv0(enh_image)
        lidar_features = self.enh_encoder.features.norm0(lidar_features)
        lidar_features = self.enh_encoder.features.relu0(lidar_features)
        lidar_features = self.enh_encoder.features.pool0(lidar_features)         
        lidar_features = self.enh_encoder.features.denseblock1(lidar_features)
        lidar_features = self.enh_encoder.features.transition1(lidar_features)

        image_embd_layer1 = self.avgpool(image_features)
        # print (image_embd_layer1.shape)
        lidar_embd_layer1 = self.avgpool(lidar_features)
        # print (lidar_embd_layer1.shape)
        image_features_layer1, lidar_features_layer1 = self.pa1(image_embd_layer1, lidar_embd_layer1)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=4, mode='bilinear', align_corners=False)
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=4, mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1
        
        # 2
        image_features = self.cxr_encoder.features.denseblock2(image_features)# output [2, 128, 28, 28]
        image_features = self.cxr_encoder.features.transition2(image_features)# output [2, 128, 28, 28] 
        lidar_features = self.enh_encoder.features.denseblock2(lidar_features) #output [2, 512, 28, 28]
        lidar_features = self.enh_encoder.features.transition2(lidar_features)
        # fusion at (B, 128, 32, 32)

        image_embd_layer2 = self.avgpool(image_features)
        lidar_embd_layer2 = self.avgpool(lidar_features)
        
        image_features_layer2, lidar_features_layer2 = self.pa2(image_embd_layer2, lidar_embd_layer2)
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=2, mode='bilinear', align_corners=False)
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=2, mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        # 3
        image_features = self.cxr_encoder.features.denseblock3(image_features) #output [2, 256, 14, 14]
        image_features = self.cxr_encoder.features.transition3(image_features)
        lidar_features = self.enh_encoder.features.denseblock3(lidar_features) #output [2, 1024, 14, 14]
        lidar_features = self.enh_encoder.features.transition3(lidar_features)
        image_embd_layer3 = self.avgpool(image_features)
        lidar_embd_layer3 = self.avgpool(lidar_features)
        
        image_features_layer3, lidar_features_layer3 = self.pa3(image_embd_layer3, lidar_embd_layer3)
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3

        # 4
        image_features = self.cxr_encoder.features.denseblock4(image_features) #output [2, 512, 7, 7]
        lidar_features = self.enh_encoder.features.denseblock4(lidar_features) #output [2, 1024, 7, 7]

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
    


class ms_transfuser_psa(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, args, config, num_classes=[1,1,1], small_dim = 256,
                 large_dim = 256, small_depth = 2, large_depth = 2, 
                 cross_attn_depth = 1, multi_scale_enc_depth = 4,
                 heads = 8, pool = 'cls', dropout = 0., emb_dropout = 0., 
                 scale_dim = 4,  weight_init=''): #config,
        super().__init__()
        
        self.encoder = Encoder(args, config)#.to(self.device)
        
        self.conv1_cxr = nn.Conv2d(1024, small_dim, kernel_size=1)
        norm_layer_cxr = partial(nn.LayerNorm, eps=1e-6)
        self.norm_cxr = norm_layer_cxr(small_dim)
        
        self.conv1_enh = nn.Conv2d(1024, large_dim, kernel_size=1)
        norm_layer_enh = partial(nn.LayerNorm, eps=1e-6)
        self.norm_enh = norm_layer_enh(large_dim)
        
        # 224 = 49 (7x7) 512 = 256(16x16) switch to the middle
        self.pos_embedding_cxr = nn.Parameter(torch.zeros(1, 49 + 1, small_dim))
        self.cls_token_cxr = nn.Parameter(torch.zeros(1, 1, small_dim))
        # print (self.cls_token_cxr.shape)
        # self.dropout_ = nn.Dropout(emb_dropout)
        # 224 = 49 (7x7) 512 = 256(16x16) switch to the middle
        self.pos_embedding_enh = nn.Parameter(torch.zeros(1, 49 + 1, large_dim))
        self.cls_token_enh = nn.Parameter(torch.zeros(1, 1, large_dim))
        
        self.dropout = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(MultiScaleTransformerEncoder(small_dim=small_dim, small_depth=small_depth,
                                                                              small_heads=heads, small_dim_head=small_dim//heads,
                                                                              small_mlp_dim=small_dim*scale_dim,
                                                                              
                                                                              large_dim=large_dim, large_depth=large_depth,
                                                                              large_heads=heads, large_dim_head=large_dim//heads,
                                                                              large_mlp_dim=large_dim*scale_dim,
                                                                              
                                                                              cross_attn_depth=cross_attn_depth, 
                                                                              cross_attn_heads=heads,
                                                                              
                                                                              dropout=dropout))

        self.pool = pool
        # self.to_latent = nn.Identity()
        self.num_classes = num_classes

        self.mlp_head_cxr = ClassificationHead(num_classes, small_dim)
        self.mlp_head_enh = ClassificationHead(num_classes, large_dim)

        self.apply(self._init_weights)

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        # head_bias = -math.log(len(self.num_classes)) if 'nlhb' in mode else 0.
        
        trunc_normal_(self.pos_embedding_cxr, std=.02)
        nn.init.normal_(self.cls_token_cxr, std=1e-6)
        
        trunc_normal_(self.pos_embedding_enh, std=.02)
        nn.init.normal_(self.cls_token_enh, std=1e-6)        
        
        # named_apply(get_init_weights_vit(mode, head_bias), self)


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
        # print (xs.shape)
        # print (xl.shape)
        
        xs = self.conv1_cxr(xs)
        xs = self.norm_cxr(xs.flatten(2).transpose(1,2), )
        # print (xs.shape)
        b, n, _ = xs.shape
        xs = torch.cat((self.cls_token_cxr.expand(b, -1, -1), xs), dim=1)
        xs = self.dropout(xs+self.pos_embedding_cxr)

        xl = self.conv1_enh(xl)
        xl = self.norm_enh(xl.flatten(2).transpose(1,2))
        # cls_token_large = repeat(self.cls_token_large, '() n d -> b n d', b=b)
        xl = torch.cat((self.cls_token_enh.expand(b, -1, -1), xl), dim=1)
        # xl += self.pos_embedding_large[:, :(n + 1)]
        xl = self.dropout(xl+self.pos_embedding_enh)        

        
        for multi_scale_transformer in self.multi_scale_transformers:
            xs, xl = multi_scale_transformer(xs, xl)
        
        xs = xs.mean(dim = 1) if self.pool == 'mean' else xs[:, 0]
        xl = xl.mean(dim = 1) if self.pool == 'mean' else xl[:, 0]

        xs = self.mlp_head_cxr(xs)
        xl = self.mlp_head_enh(xl)
        
        logits = list()
        # # [(N, H, W), (N, H, W),...]
        # logit_maps = list()
        for index, num_class in enumerate(self.num_classes):
            # if self.cfg.attention_map != "None":
            #     feat_map = self.attention_map(feat_map)

            # classifier = getattr(self, "fc_" + str(index))

            # logit = classifier(x)
            # (N, num_class)
            # logit = logit.squeeze(-1).squeeze(-1)
            # print(xs)
            # print (xl)
            logits.append(xs[index]+xl[index])
            # print (logits)
        # return (logits, logit_maps)
        return logits