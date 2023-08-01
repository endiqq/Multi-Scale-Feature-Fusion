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



#%%
class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, args, config):
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

        elif args.network == 'res34_dense121':
            
            self.cxr_encoder = models.resnet34(pretrained=True)
            ftrs_vector_cxr = self.cxr_encoder.fc.in_features
            
            # model_cxr = models.resnet50(pretrained=True)
            model_enh = models.densenet121(pretrained=True)
            ftrs_vector_enh = model_enh.classifier.in_features
            # self.conv1_enh = nn.Conv(1024, config.n_embd, kernel_size=1)
            self.enh_encoder = model_enh.features
            
        elif args.network == 'res18_res34':
            
            self.cxr_encoder = models.resnet18(pretrained=True)
            ftrs_vector = self.cxr_encoder.fc.in_features
            
            self.enh_encoder = models.resnet34(pretrained=True)
            # ftrs_vector_enh = self.enh_encoder.fc.in_features

            
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
        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        

        self.transformer1 = GPT(n_embd=64,
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
        self.transformer2 = GPT(n_embd=128,
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
        self.transformer3 = GPT(n_embd=256,
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
        # image_features = self.cxr_encoder.transition1(image_features)
        
        # lidar_features = self.enh_encoder.denseblock1(lidar_features)
        # lidar_features = self.enh_encoder.transition1(lidar_features)
        
        # fusion at (B, 64, 64, 64)
        image_embd_layer1 = self.avgpool(image_features) # (bz, 64, 56, 56)
        lidar_embd_layer1 = self.avgpool(lidar_features) # (bz, 128, 28, 28)
        
        # bz, small_dim, _, _ = image_embd_layer1.shape
        # bz, large_dim, _, _ = lidar_embed_layer1.shape
        
        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=8, mode='bilinear', align_corners=False)
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=8, mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1

        # 2
        image_features = self.cxr_encoder.layer2(image_features)#[2, 128, 28, 28]
        lidar_features = self.enh_encoder.layer2(lidar_features)
        # lidar_features = self.enh_encoder.transition2(lidar_features)#[2, 256, 14, 14]
        # fusion at (B, 128, 32, 32)
        image_embd_layer2 = self.avgpool(image_features)#[2, 128, 14, 14]
        lidar_embd_layer2 = self.avgpool(lidar_features)#[2, 256, 7, 7]
        
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2)
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=4, mode='bilinear', align_corners=False)
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=4, mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        #3
        image_features = self.cxr_encoder.layer3(image_features)
        lidar_features = self.enh_encoder.layer3(lidar_features)
        # lidar_features = self.enh_encoder.transition3(lidar_features)#[2, 256, 14, 14]
        # fusion at (B, 256, 16, 16)
        image_embd_layer3 = self.avgpool(image_features)
        lidar_embd_layer3 = self.avgpool(lidar_features)
        
        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3)
        image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=2, mode='bilinear', align_corners=False)
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, scale_factor=2, mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3

        #4 
        image_features = self.cxr_encoder.layer4(image_features)
        lidar_features = self.enh_encoder.layer4(lidar_features)
        # lidar_features = self.enh_encoder.transition4(lidar_features)#[2, 256, 14, 14]


        # fusion at (B, 512, 8, 8)
        image_embd_layer4 = self.avgpool(image_features)
        lidar_embd_layer4 = self.avgpool(lidar_features)
        
        
        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4)
        # image_features_layer4 = self.avgpool_enh(image_features_layer4)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4
        

        image_features = self.cxr_encoder.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)
        lidar_features = self.enh_encoder.avgpool(lidar_features)
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
#%%
class ClassificationHead(nn.Module):
    def __init__(self, num_classes, large_dim, extend=True):
        super(ClassificationHead, self).__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(large_dim, eps=1e-6)
        # self.embed_dim = large_dim
        
        for index, num_class in enumerate(num_classes):
            if extend:
                setattr(self, "fc_" + str(index), nn.Sequential(nn.Linear(512, 256),
                                                                nn.ReLU(inplace=True),
                                                                nn.Linear(256, 128),
                                                                nn.ReLU(inplace=True),
                                                                nn.Linear(128, 64),
                                                                nn.ReLU(inplace=True),
                                                                nn.Linear(64, num_class)))
            else:
                setattr(self, "fc_" + str(index), nn.Linear(large_dim, num_class))

    def forward(self, x):
        feat_map = self.norm(x)
        logits = list()
        for index, num_class in enumerate(self.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            logit = classifier(feat_map)
            logits.append(logit)

        return logits


class ms_transfuser(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, args, config, num_classes): 
        super().__init__()
        
        self.num_classes = num_classes
        self.config = config
        # self.pred_len = config.pred_len
        # self.cfg = cfg
        # self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        # self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        
        if args.network =='Res50':
            ftrs_vector = 2048            
        elif args.network == 'dense121' or args.network == 'chexpert':
            ftrs_vector = 1024
        elif args.network == 'res34_dense121':
            ftrs_vector_l = 1024; ftrs_vector_s = 512
        elif args.network == 'res18_res34':
            ftrs_vector = 512
        

        self.encoder = Encoder(args, config)#.to(self.device)
        

            # self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        self.output = ClassificationHead(num_classes, ftrs_vector) #nn.Linear(64, 2).to(self.device)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, image_list, lidar_list):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
        '''
        z = self.encoder(image_list, lidar_list)
        # if self.extend == True:
        #     z = self.join(fused_features)
        # else:
        #     z = fused_features
    
        logits = self.output(z)
        # logits_large = self.output_large(large_features)
        # logits = list()
        # for idx in range(len(self.num_classes)):
        #     logits.append(logits_small[idx]+logits_large[idx])
        
        return logits

