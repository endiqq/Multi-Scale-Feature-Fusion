#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 23:56:24 2020

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




class Classifier(nn.Module):

    def __init__(self, num_classes, model_name):
        super(Classifier, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name
        self._init_classifier()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if self.model_name == 'Res50':
            model = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(model.children()))[:-1]
        elif self.model_name == 'vit_small':
            model = timm.create_model("vit_small_patch16_224", pretrained=True)
            self.backbone = nn.Sequential(*list(model.children()))[:-3]
            self.fc_norm = norm_layer(384)
            self.pre_logits = nn.Identity()
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(model_name)
            )
            
        
    def _init_classifier(self):

        for index, num_class in enumerate(self.num_classes):
            if self.model_name == 'Res50':
                setattr(self, "fc_" + str(index),
                    nn.Linear(1024*2, num_class))
            elif self.model_name == 'vit_small':
                setattr(self, "fc_" + str(index),
                    nn.Linear(384, num_class))                

        classifier = getattr(self, "fc_" + str(index))
        
        if isinstance(classifier, nn.Linear):
            classifier.weight.data.normal_(0, 0.01)
            classifier.bias.data.zero_()


    def forward(self, x):
        # (N, C, H, W)
        feat = self.backbone(x)
        if self.model_name == 'Res50':
            x = feat.squeeze(-1).squeeze(-1)
        elif self.model_name == 'vit_small':
            x = feat[:, 0]
            x = self.fc_norm(x)
            x = self.pre_logits(x)
            
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


# class Vgg16bn_bn_conv(nn.Module):
#     def __init__(self, num_class):
#         super(Vgg16bn_bn_conv, self).__init__()
#         model = torchvision.models.vgg16_bn(pretrained = True)
#         self.features = model.features[0:-1]
                
#         self.classifier = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
#                                         nn.BatchNorm2d(256),
#                                         nn.ReLU(inplace=True),
#                                         nn.Conv2d(in_channels=256, out_channels = num_class, kernel_size=1),
#                                         nn.BatchNorm2d(num_class),
#                                         nn.ReLU(inplace=True))      

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         x = F.avg_pool2d(x, (x.shape[-2], x.shape[-1])).squeeze(2).squeeze(2)
#         return x


# class ConvStem(nn.Module):

#     def __init__(self, model_name, embed_dim=256, norm_layer=None):
#         super(ConvStem, self).__init__()

#         # self.num_classes = num_classes
#         self.model_name = model_name
#         # self._init_classifier()
#         # norm_layer = partial(nn.LayerNorm, eps=1e-6)
#         if self.model_name == 'Res50':
#             self.conv1 = nn.Conv2d(2048, 256, kernel_size=1)
#         elif self.model_name == 'dense121':
#             self.conv1 = nn.Conv2d(1024, 256, kernel_size=1)
        
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
#         if self.model_name == 'Res50':
#             model = torchvision.models.resnet50(pretrained=True)
#             self.backbone = nn.Sequential(*list(model.children()))[:-2]
#         elif self.model_name == 'dense121':         
#             model = torchvision.models.densenet121(pretrained=True)
#             self.backbone = model.features
#         # elif self.model_name == 'vit_small':
#         #     model = timm.create_model("vit_small_patch16_224", pretrained=True)
#         #     self.backbone = nn.Sequential(*list(model.children()))[:-3]
#         #     self.fc_norm = norm_layer(384)
#         #     self.pre_logits = nn.Identity()
#         else:
#             raise Exception(
#                 'Unknown backbone type : {}'.format(model_name)
#             )
            

#     def forward(self, x):
#         # (N, C, H, W)
#         x = self.backbone(x)
#         x = self.conv1(x)
#         x = x.flatten(2).transpose(1,2)
        
#         return x
        
# def res50_vit(**kwargs):
#     # minus one ViT block
#     model = VisionTransformer(
#         patch_size=16, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem('Res50'), **kwargs)
#     model.default_cfg = _cfg()
#     return model
    
