#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:17:20 2022

@author: endiqq
"""

from efficientnet_pytorch import EfficientNet
import pretrainedmodels
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import timm
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers.helpers import to_2tuple
import torch
import json
from easydict import EasyDict as edict
import os

# import torch
from torch import nn, einsum
import torch.nn.functional as F
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


class ConvStem(nn.Module):

    def __init__(self, model_name, embed_dim=256, norm_layer=None):
        super(ConvStem, self).__init__()

        # self.num_classes = num_classes
        self.model_name = model_name
        # self._init_classifier()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        
        if self.model_name == 'Res50':
            self.conv1 = nn.Conv2d(2048, embed_dim, kernel_size=1)
        elif self.model_name == 'dense121' or self.model_name == 'chexpert':
            self.conv1 = nn.Conv2d(1024, embed_dim, kernel_size=1)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
        if self.model_name == 'Res50':
            model = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(model.children()))[:-2]
        elif self.model_name == 'dense121':         
            model = torchvision.models.densenet121(pretrained=True)
            self.backbone = model.features
        elif self.model_name == 'chexpert':
            
            model = Classifier(cfg)
            model_path = 'CheXpert_logdir'
            ckpt_path = os.path.join(model_path, 'best3_CXR.ckpt')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
            self.backbone = model.backbone
            
            # model_enh = Classifier(cfg)
            # model_enh_path = '../CheXpert_logdir'
            # ckpt_path = os.path.join(model_enh_path, 'best2_Enh.ckpt')
            # ckpt = torch.load(ckpt_path, map_location='cpu')
            # model_enh.load_state_dict(ckpt['state_dict'])
            # self.model_enh = model_enh
        # elif self.model_name == 'vit_small':
        #     model = timm.create_model("vit_small_patch16_224", pretrained=True)
        #     self.backbone = nn.Sequential(*list(model.children()))[:-3]
        #     self.fc_norm = norm_layer(384)
        #     self.pre_logits = nn.Identity()
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(model_name)
            )         

    def forward(self, x):
        # (N, C, H, W)
        x = self.backbone(x)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        
        return x



class ConvStem_Enh(nn.Module):

    def __init__(self, model_name, embed_dim=256, norm_layer=None):
        super(ConvStem_Enh, self).__init__()

        # self.num_classes = num_classes
        self.model_name = model_name
        # self._init_classifier()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        
        if self.model_name == 'Res50':
            self.conv1 = nn.Conv2d(2048, embed_dim, kernel_size=1)
        elif self.model_name == 'dense121' or self.model_name == 'chexpert':
            self.conv1 = nn.Conv2d(1024, embed_dim, kernel_size=1)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
        if self.model_name == 'Res50':
            model = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(model.children()))[:-2]
        elif self.model_name == 'dense121':         
            model = torchvision.models.densenet121(pretrained=True)
            self.backbone = model.features
        elif self.model_name == 'chexpert':
            
            # model = Classifier(cfg)
            # model_path = '../CheXpert_logdir'
            # ckpt_path = os.path.join(model_cxr_path, 'best3_CXR.ckpt')
            # ckpt = torch.load(ckpt_path, map_location='cpu')
            # model.load_state_dict(ckpt['state_dict'])
            # self.backbone = model_cxr.backbone
            
            model_enh = Classifier(cfg)
            model_enh_path = 'CheXpert_logdir'
            ckpt_path = os.path.join(model_enh_path, 'best2_Enh.ckpt')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model_enh.load_state_dict(ckpt['state_dict'])
            self.backbone = model_enh.backbone
        # elif self.model_name == 'vit_small':
        #     model = timm.create_model("vit_small_patch16_224", pretrained=True)
        #     self.backbone = nn.Sequential(*list(model.children()))[:-3]
        #     self.fc_norm = norm_layer(384)
        #     self.pre_logits = nn.Identity()
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(model_name)
            )         

    def forward(self, x):
        # (N, C, H, W)
        x = self.backbone(x)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        # print (x.shape)
        return x


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
    
# vit_b = 768; vit_s = 384    
class Fus_CrossViT(nn.Module):
        # img_size = 512; small_dim=large_dim=256; img_size = 224; small_dim=large_dim=49
    def __init__(self, model_name, num_classes=[1,1,1], small_dim = 256,
                 large_dim = 256, small_depth = 2, large_depth = 2, 
                 cross_attn_depth = 1, multi_scale_enc_depth = 4,
                 heads = 8, pool = 'cls', dropout = 0., emb_dropout = 0., 
                 scale_dim = 4,  weight_init=''):
        
        super().__init__()
        
        # self.cfg = cfg
        # self.patch_embed_cxr = model_vit_cxr.features3D
        # self.vit_cxr = model_vit_cxr
        
        # self.patch_embed_enh = model_vit_enh.features3D
        # self.vit_enh = model_vit_enh
        if model_name == 'chexpert':
            self.patch_embed_cxr = ConvStem(model_name, embed_dim = small_dim)
            self.patch_embed_enh = ConvStem_Enh(model_name, embed_dim = large_dim)               
        else:
            self.patch_embed_cxr = ConvStem(model_name, embed_dim = small_dim)
            self.patch_embed_enh = ConvStem(model_name, embed_dim = large_dim)        

        # if model_name == 'chexpert':
        #     self.patch_embed_cxr = ConvStem(model_name, embed_dim = 256)
        #     self.patch_embed_enh = ConvStem_Enh(model_name, embed_dim = 256)               
        # else:
        #     self.patch_embed_cxr = ConvStem(model_name, embed_dim = 256)
        #     self.patch_embed_enh = ConvStem(model_name, embed_dim = 256)  

        self.pos_embedding_cxr = nn.Parameter(torch.zeros(1, 256 + 1, small_dim))
        self.cls_token_cxr = nn.Parameter(torch.zeros(1, 1, small_dim))
        # print (self.cls_token_cxr.shape)
        # self.dropout_ = nn.Dropout(emb_dropout)

        self.pos_embedding_enh = nn.Parameter(torch.zeros(1, 256 + 1, large_dim))
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
        
        # cxr_ftrs = self.vit_features_cxr(img_cxr) # b, 197, 384
        # bs, n, dim = cxr_ftrs.shape
        # # xs = xs[:,0] # b, 384
        # x_cxr = vit_cxr(img_cxr)
        
        # enh_ftrs = self.vit_features_enh(img_enh)
        # # print (xl.shape)
        # x_enh = vit_enh(img_enh)
        
        xs = self.patch_embed_cxr(img_cxr)
        # print (xs.shape)
        b, n, _ = xs.shape
        # print (xs.shape)
        # cls_token_small = repeat(self.cls_token_small, '() n d -> b n d', b = b)
        # print (self.cls_token_cxr.expand(b, -1, -1).shape)
        xs = torch.cat((self.cls_token_cxr.expand(b, -1, -1), xs), dim=1)

        # print (xs.shape)
        # xs += self.pos_embedding_small[:, :(n + 1)]
        xs = self.dropout(xs+self.pos_embedding_cxr)

        xl = self.patch_embed_enh(img_enh)
        b, n, _ = xl.shape

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
        
        # return x
    
    
    
# class Cross_Attention(nn.Module):
    
#     def __init__(self, model_name, num_classes, embed_dim=256, depth=4, num_heads=8):
#         super(Cross_Attention, self).__init__()
    
#         # minus one ViT block
#         model = VisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads,\
#             norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
#         self.patch_embed_cxr = ConvStem(model_name, embed_dim = embed_dim)
#         self.patch_embed_enh = ConvStem(model_name, embed_dim = embed_dim)
        
#         self.blocks = model.blocks
        
#         self.norm = model.norm
#         self.num_tokens = 1
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, 256 + self.num_tokens, embed_dim))
#         self.pos_drop = nn.Dropout(0.0)
        
#         # deficlassifer
#         self.model_name = model_name
#         self.num_classes = num_classes
#         self._init_classifier()

        
    
#     def _init_classifier(self):

#         for index, num_class in enumerate(self.num_classes):
#             if self.model_name == 'Res50' or self.model_name == 'dense121':
#                 setattr(self, "fc_" + str(index),
#                     nn.Linear(256, num_class))
#             elif self.model_name == 'vit_small':
#                 setattr(self, "fc_" + str(index),
#                     nn.Linear(384, num_class))                

#         classifier = getattr(self, "fc_" + str(index))
        
#         if isinstance(classifier, nn.Linear):
#             classifier.weight.data.normal_(0, 0.01)
#             classifier.bias.data.zero_()
    
#     def forward_features(self, cxr, enh):
        
#         x = self.patch_embed(cxr)
#         x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
#         x = self.pos_drop(x + self.pos_embed)
        
#         y = self.patch_embed(enh)
#         y = torch.cat((self.cls_token.expand(y.shape[0], -1, -1), y), dim=1)
#         y = self.pos_drop(y + self.pos_embed)        
#         # if self.grad_checkpointing and not torch.jit.is_scripting():
#         #     x = checkpoint_seq(self.blocks, x)
#         # else:
#         fus = self.blocks(x, y)
#         fus = self.norm(fus)
#         return fus

#     def forward(self, cxr, enh):
#         # (N, C, H, W)
#         feat = self.forward_features(cxr, enh)
#         x = feat[:,0]
#         # if self.model_name == 'Res50':
#         #     x = feat.squeeze(-1).squeeze(-1)
#         # elif self.model_name == 'vit_small':
#         #     x = feat[:, 0]
#         #     x = self.fc_norm(x)
#         #     x = self.pre_logits(x)
            
#         # print (feat.shape)
#         # # [(N, 1), (N,1),...]
#         logits = list()
#         # # [(N, H, W), (N, H, W),...]
#         # logit_maps = list()
#         for index, num_class in enumerate(self.num_classes):
#             # if self.cfg.attention_map != "None":
#             #     feat_map = self.attention_map(feat_map)

#             classifier = getattr(self, "fc_" + str(index))

#             logit = classifier(x)
#             # (N, num_class)
#             # logit = logit.squeeze(-1).squeeze(-1)
#             logits.append(logit)

#         # return (logits, logit_maps)
#         return logits