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

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models

from model.classifier import Classifier  # noqa


# print (args.cfg_path)
with open('covid_vit/config/example_PCAM_xq.json') as f:
    cfg = edict(json.load(f))
    # if args.verbose is True:
    #     print(json.dumps(cfg, indent=4))

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, eps=1e-06)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-06)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


#%%

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        
        self.n_embd = n_embd #input feature
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, (self.config.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))
        # config.n_views = seq_len = 1; vert_anchor=horz_achor = 16 (input size = 512) 
        # velocity embedding
        # self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop) # embd_pdrop=0.0; original=0.1

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd, eps=1e-06)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def configure_optimizers(self):
    #     # separate out all parameters to those that will and won't experience regularizing weight decay
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
    #     for mn, m in self.named_modules():
    #         for pn, p in m.named_parameters():
    #             fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

    #             if pn.endswith('bias'):
    #                 # all biases will not be decayed
    #                 no_decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
    #                 # weights of whitelist modules will be weight decayed
    #                 decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
    #                 # weights of blacklist modules will NOT be weight decayed
    #                 no_decay.add(fpn)

    #     # special case the position embedding parameter in the root GPT module as not decayed
    #     no_decay.add('pos_emb')

    #     # create the pytorch optimizer object
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     optim_groups = [
    #         {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
    #         {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    #     ]

    #     return optim_groups

    def forward(self, cxr_tensor, enh_tensor):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        
        bz = cxr_tensor.shape[0]
        h, w = cxr_tensor.shape[2:4]
        
        # forward the image model for token embeddings
        image_tensor = cxr_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor = enh_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1).permute(0,1,3,4,2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)

        # # project velocity to n_embed
        # velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings) # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        x = x.view(bz, (self.config.n_views + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings

        image_tensor_out = x[:, :self.config.n_views*self.seq_len, :, :, :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.config.n_views*self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        
        return image_tensor_out, lidar_tensor_out


class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config, args):
        super().__init__()
        
        # if cfg.finetune == False:
        #     for params in model_cxr.parameters():
        #         params.requires_grad = False
                
        #     for params in model_enh.parameters():
        #         params.requires_grad = False

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
            
             
        self.config = config
        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors)) # orignal
        # self.avgpool_cxr = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors)) # changed but never used
        # self.avgpool_enh = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors)) 
        #6/29 change to avgpool_cxr and avgpool_enh
        # self.cxr_encoder = model_cxr.model.features
        # self.enh_encoder = model_enh.model.features
        

        self.transformer4 = GPT(n_embd=ftrs_vector,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)

        
    def forward(self, cxr_image, enh_image):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''
        # # image normalization
        # if self.image_encoder.normalize:
        #     image_list = [normalize_imagenet(image_input) for image_input in image_list]

        bz, _, h, w = cxr_image.shape
        
        # img_channel = cxr_image.shape[1]
        # lidar_channel = enh_image.shape[1]
        
        # self.config.n_views = len(image_list) // self.config.seq_len

        # image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
        # lidar_tensor = torch.stack(lidar_list, dim=1).view(bz * self.config.seq_len, lidar_channel, h, w)

        image_features = self.cxr_encoder(cxr_image)
        # image_features = self.image_encoder.features.bn1(image_features)
        # image_features = self.image_encoder.features.relu(image_features)
        # image_features = self.image_encoder.features.maxpool(image_features)
        
        lidar_features = self.enh_encoder(enh_image)
        # lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        # lidar_features = self.lidar_encoder._model.relu(lidar_features)
        # lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        
        # # fusion at (B, 512, 8, 8)
        # image_embd_layer4 = self.avgpool(image_features)
        # lidar_embd_layer4 = self.avgpool(lidar_features)
        
        # # remove dimension to n_embd
        # image_embd_layer4 = self.conv1_cxr(image_features)
        # lidar_embd_layer4 = self.conv1_enh(lidar_features)
        
        
        # fusion at (B, 512, 16, 16)
        image_embd_layer4 = self.avgpool(image_features)
        lidar_embd_layer4 = self.avgpool(lidar_features)
        # image_embd_layer4 = self.avgpool_cxr(image_features)
        # lidar_embd_layer4 = self.avgpool_enh(lidar_features)
        
        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4)
        
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4
        
        image_features = F.relu(image_features, inplace=True)
        image_features = F.adaptive_avg_pool2d(image_features, (1, 1))
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)
        
        lidar_features = F.relu(lidar_features, inplace=True)
        lidar_features = F.adaptive_avg_pool2d(lidar_features, (1, 1))
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = lidar_features.view(bz, self.config.seq_len, -1)

        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)

        return fused_features

# class PIDController(object):
#     def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
#         self._K_P = K_P
#         self._K_I = K_I
#         self._K_D = K_D

#         self._window = deque([0 for _ in range(n)], maxlen=n)
#         self._max = 0.0
#         self._min = 0.0

#     def step(self, error):
#         self._window.append(error)
#         self._max = max(self._max, abs(error))
#         self._min = -abs(self._max)

#         if len(self._window) >= 2:
#             integral = np.mean(self._window)
#             derivative = (self._window[-1] - self._window[-2])
#         else:
#             integral = 0.0
#             derivative = 0.0

#         return self._K_P * error + self._K_I * integral + self._K_D * derivative

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

class Fus_Attention(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, args, config, num_classes, extend=False):
        super().__init__()
        
        self.extend = extend
        self.config = config
        # self.pred_len = config.pred_len
        # self.cfg = cfg
        # self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        # self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        
        if args.network =='Res50':
            ftrs_vector = 2048            
        elif args.network == 'dense121' or args.network == 'chexpert':
            ftrs_vector = 1024
        

        self.encoder = Encoder(config, args)#.to(self.device)
        
        if self.extend == True:
            self.join = nn.Sequential(
                                nn.Linear(1024, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 128),
                                nn.ReLU(inplace=True),
                            )#.to(self.device)
            
        
            # self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
            self.output = ClassificationHead(num_classes, 128) #nn.Linear(64, 2).to(self.device)
        else:
            self.output = ClassificationHead(num_classes, ftrs_vector) #nn.Linear(64, 2).to(self.device)
        
    def forward(self, image_list, lidar_list):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
        '''
        fused_features = self.encoder(image_list, lidar_list)
        if self.extend == True:
            z = self.join(fused_features)
        else:
            z = fused_features
    
        logits = self.output(z)

        # output_wp = list()
        
        return logits

        # # initial input variable to GRU
        # x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # # autoregressive generation of output waypoints
        # for _ in range(self.pred_len):
        #     # x_in = torch.cat([x, target_point], dim=1)
        #     x_in = x + target_point
        #     z = self.decoder(x_in, z)
        #     dx = self.output(z)
        #     x = dx + x
        #     output_wp.append(x)

        # pred_wp = torch.stack(output_wp, dim=1)

        # return pred_wp

    # def control_pid(self, waypoints, velocity):
    #     ''' 
    #     Predicts vehicle control with a PID controller.
    #     Args:
    #         waypoints (tensor): predicted waypoints
    #         velocity (tensor): speedometer input
    #     '''
    #     assert(waypoints.size(0)==1)
    #     waypoints = waypoints[0].data.cpu().numpy()

    #     # flip y is (forward is negative in our waypoints)
    #     waypoints[:,1] *= -1
    #     speed = velocity[0].data.cpu().numpy()

    #     desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
    #     brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

    #     aim = (waypoints[1] + waypoints[0]) / 2.0
    #     angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
    #     if(speed < 0.01):
    #         angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
    #     steer = self.turn_controller.step(angle)
    #     steer = np.clip(steer, -1.0, 1.0)

    #     delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
    #     throttle = self.speed_controller.step(delta)
    #     throttle = np.clip(throttle, 0.0, self.config.max_throttle)
    #     throttle = throttle if not brake else 0.0

    #     metadata = {
    #         'speed': float(speed.astype(np.float64)),
    #         'steer': float(steer),
    #         'throttle': float(throttle),
    #         'brake': float(brake),
    #         'wp_2': tuple(waypoints[1].astype(np.float64)),
    #         'wp_1': tuple(waypoints[0].astype(np.float64)),
    #         'desired_speed': float(desired_speed.astype(np.float64)),
    #         'angle': float(angle.astype(np.float64)),
    #         'aim': tuple(aim.astype(np.float64)),
    #         'delta': float(delta.astype(np.float64)),
    #     }

    #     return steer, throttle, brake, metadata
