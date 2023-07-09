#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:22:45 2022

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

from model.classifier import Classifier  # noqa

# print (args.cfg_path)
with open('covid_vit/config/example_PCAM_xq.json') as f:
    cfg = edict(json.load(f))
    # if args.verbose is True:
    #     print(json.dumps(cfg, indent=4))

# __all__ = [
#     'res50_vit',
# ]

# class ConvStem(nn.Module):

#     def __init__(self, model_name, img_size=224, patch_size=16, in_chans=3, \
#                  embed_dim=256, norm_layer=None, flatten=True):
#         super(ConvStem, self).__init__()
        
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten
        

#         # build stem
#         # self.model_name = model_name
#         # self._init_classifier()
#         # norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         # self.norm = norm_layer(embed_dim)
#         if model_name == 'Res50':
#             conv1 = nn.Conv2d(2048, 256, kernel_size=1)
#         elif model_name == 'dense121':
#             conv1 = nn.Conv2d(1024, 256, kernel_size=1)
    
#         if model_name == 'Res50':
#             model = torchvision.models.resnet50(pretrained=True)
#             backbone = nn.Sequential(*list(model.children()))[:-2]
#         elif model_name == 'dense121':         
#             model = torchvision.models.densenet121(pretrained=True)
#             backbone = model.features
#         # elif self.model_name == 'vit_small':
#         #     model = timm.create_model("vit_small_patch16_224", pretrained=True)
#         #     self.backbone = nn.Sequential(*list(model.children()))[:-3]
#         #     self.fc_norm = norm_layer(384)
#         #     self.pre_logits = nn.Identity()
#         else:
#             raise Exception(
#                 'Unknown backbone type : {}'.format(model_name)
#             )
        
#         self.proj = nn.Sequential(backbone,
#                                    conv1,)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x

# model = VisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads, \
#                           norm_layer=partial(nn.LayerNorm, eps=1e-6))
# model.default_cfg = _cfg()        



class ConvStem(nn.Module):

    def __init__(self, model_name, data_type, embed_dim=256, norm_layer=None):
        super(ConvStem, self).__init__()

        # self.num_classes = num_classes
        self.model_name = model_name
        # self._init_classifier()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        
        if self.model_name == 'Res50':
            self.conv1 = nn.Conv2d(2048, 256, kernel_size=1)
        elif self.model_name == 'dense121' or self.model_name == 'chexpert':
            self.conv1 = nn.Conv2d(1024, 256, kernel_size=1)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
        if self.model_name == 'Res50':
            model = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(model.children()))[:-2]
        elif self.model_name == 'dense121':         
            model = torchvision.models.densenet121(pretrained=True)
            self.backbone = model.features
        elif args.network == 'chexpert':
            model = Classifier(cfg)
            model_path = 'CheXpert_logdir'
            
            if data_type == 'data':
                ckpt_path = os.path.join(model_path, 'best3_CXR.ckpt')         
            elif data_type == 'Train_Mix':
                ckpt_path = os.path.join(model_path, 'best2_Enh.ckpt')
                
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
            self.backbone = model_cxr.backbone.features

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


class convstem_vit(nn.Module):
    
    def __init__(self, model_name, data_type, num_classes, embed_dim=256, depth=4, num_heads=8):
        super(convstem_vit, self).__init__()
    
        # minus one ViT block
        model = VisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads,\
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.patch_embed = ConvStem(model_name, data_type, embed_dim = embed_dim)
        self.blocks = model.blocks
        self.norm = model.norm
        self.num_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 256 + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        
        # deficlassifer
        self.model_name = model_name
        self.num_classes = num_classes
        self._init_classifier()

        
    
    def _init_classifier(self):

        for index, num_class in enumerate(self.num_classes):
            if self.model_name == 'Res50' or self.model_name == 'dense121':
                setattr(self, "fc_" + str(index),
                    nn.Linear(256, num_class))
            elif self.model_name == 'vit_small':
                setattr(self, "fc_" + str(index),
                    nn.Linear(384, num_class))                

        classifier = getattr(self, "fc_" + str(index))
        
        if isinstance(classifier, nn.Linear):
            classifier.weight.data.normal_(0, 0.01)
            classifier.bias.data.zero_()
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        # (N, C, H, W)
        feat = self.forward_features(x)
        x = feat[:,0]
        # if self.model_name == 'Res50':
        #     x = feat.squeeze(-1).squeeze(-1)
        # elif self.model_name == 'vit_small':
        #     x = feat[:, 0]
        #     x = self.fc_norm(x)
        #     x = self.pre_logits(x)
            
        # print (feat.shape)
        # # [(N, 1), (N,1),...]
        logits = list()
        # # [(N, H, W), (N, H, W),...]
        # logit_maps = list()
        for index, num_class in enumerate(self.num_classes):
            # if self.cfg.attention_map != "None":
            #     feat_map = self.attention_map(feat_map)

            classifier = getattr(self, "fc_" + str(index))

            logit = classifier(x)
            # (N, num_class)
            # logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)

        # return (logits, logit_maps)
        return logits

    
# model = VisionTransformer(
#     patch_size=16, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
#     norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=partial(ConvStem, 'Res50'))
# model.default_cfg = _cfg()