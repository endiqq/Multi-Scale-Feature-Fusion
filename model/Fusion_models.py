#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 00:10:52 2020

@author: endiqq
"""


import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Late_Fusion_Net(nn.Module):
    def __init__(self, NN, my_model1, my_model2, num_classes):
        super(Late_Fusion_Net, self).__init__()        
        self.NN = NN
        self.num_classes = num_classes

        if self.NN == 'Res50': 
                self.feature1 = nn.Sequential(*list(my_model1.children())[:-1])
                self.feature2 = nn.Sequential(*list(my_model2.children())[:-1])
                                
                self.classifier1 = my_model1.fc
                self.classifier2 = my_model2.fc
                
        elif self.NN == 'xception' or self.NN == 'inceptionresnetv2':
                self.feature1 = nn.Sequential(*list(my_model1.children())[:-1])
                self.feature2 = nn.Sequential(*list(my_model2.children())[:-1])
                                
                self.classifier1 = my_model1.last_linear
                self.classifier2 = my_model2.last_linear
                        
        elif self.NN == 'efficientnet-b4':
                self.feature1 = my_model1
                self.feature2 = my_model2
                
                self.avgpool =  my_model1._avg_pooling
                self.dropout = my_model1._dropout
                self.swish = my_model1._swish
                
                self.classifier1 = my_model1._fc
                self.classifier2 = my_model2._fc
                        
        elif self.NN == 'Alexnet' or self.NN == 'Vgg16bn':
            self.feature1 = my_model1.features
            self.feature2 = my_model2.features
            
            self.avgpool = my_model1.avgpool
            
            self.classifier1 = my_model1.classifier
            self.classifier2 = my_model2.classifier
            
        elif self.NN =='Vgg16bn_bn_conv':
            self.feature1 = my_model1.features
            self.feature2 = my_model2.features            
            
            self.classifier1 = my_model1.classifier
            self.classifier2 = my_model2.classifier
                                    
    def forward(self,x,y):       
        if self.NN == 'efficientnet-b4':
            x1 = self.feature1.extract_features(x)            
            x2 = self.feature2.extract_features(y)
        else:   
            x1 = self.feature1(x)            
            x2 = self.feature2(y)
        
        if (self.NN == 'Alexnet' or self.NN == 'Vgg16' or self.NN == "Vgg16bn"
            or self.NN == 'Vgg19' or self.NN == 'Vgg19bn'):
            
            x1 = self.avgpool(x1)
            x1 = x1.view(x1.size(0),-1)
            
            x2 = self.avgpool(x2)
            x2 = x2.view(x2.size(0),-1)
                        
            x1 = self.classifier1(x1) 
            x2 = self.classifier2(x2)            
        
        elif self.NN == 'Vgg16bn_bn_conv':
            
            x1 = self.classifier1(x1)
            x1 = F.avg_pool2d(x1, (x1.shape[-2], x1.shape[-1])).squeeze(2).squeeze(2)
            x2 = self.classifier2(x2)
            x2 = F.avg_pool2d(x2, (x2.shape[-2], x2.shape[-1])).squeeze(2).squeeze(2)             
        
        elif self.NN == 'xception':
            
            x1 = F.adaptive_avg_pool2d(x1, (1, 1))
            x1 = x1.view(x1.size(0), -1)
            
            x2 = F.adaptive_avg_pool2d(x2, (1, 1))
            x2 = x2.view(x2.size(0), -1)
        
            x1 = self.classifier1(x1)             
            x2 = self.classifier2(x2)
            
        elif self.NN == 'inceptionresnetv2':

            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            
            x1 = self.classifier1(x1)                
            x2 = self.classifier2(x2)
            
        elif self.NN == 'efficientnet-b4':
            
            x1 = self.dropout(self.avgpool(x1))
            x1 = x1.view(x1.size(0), -1)
            x1 = self.swish(self.classifier1(x1))
            
            x2 = self.dropout(self.avgpool(x2))
            x2 = x2.view(x2.size(0), -1)
            x2 = self.swish(self.classifier2(x2))
                        
        else: # Res50 

            x1 = x1.view(x1.size(0),-1)                      
            x2 = x2.view(x2.size(0),-1)

            x1 = self.classifier1(x1) 
            x2 = self.classifier2(x2)  
            
        x3 = x1+x2
            
        return x3
    
class Mid_Fusion_Net(nn.Module):
    def __init__(self, NN, my_model1, my_model2, embedding_dim, num_classes, mtd):
        super(Mid_Fusion_Net, self).__init__()        
        self.NN = NN
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.mtd = mtd
        
        if self.NN == 'Alexnet':
            self.fusion_conv = nn.Sequential(nn.Conv2d(2*256, 256, kernel_size=1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(inplace=True),)
        else:
            self.fusion_conv = nn.Sequential(nn.Conv2d(2*self.embedding_dim, self.embedding_dim, kernel_size=1),
                                             nn.BatchNorm2d(self.embedding_dim),
                                             nn.ReLU(inplace=True),)
        
        self.fusion_concat = nn.Sequential(nn.Linear(2*self.embedding_dim, self.embedding_dim),
                                       nn.ReLU(inplace = True),
                                       nn.Dropout(p=0.2))
        
        if self.NN == 'Res50': 
                self.feature1 = nn.Sequential(*list(my_model1.children())[:-2])
                self.feature2 = nn.Sequential(*list(my_model2.children())[:-2])
                
                self.avgpool = my_model1.avgpool

                self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
                
        elif self.NN == 'xception':
                self.feature1 = nn.Sequential(*list(my_model1.children())[:-1])
                self.feature2 = nn.Sequential(*list(my_model2.children())[:-1])
                                
                self.classifier =  nn.Linear(2048, self.num_classes)
        
        elif self.NN == 'inceptionresnetv2':
                self.feature1 = nn.Sequential(*list(my_model1.children())[:-2])
                self.feature2 = nn.Sequential(*list(my_model2.children())[:-2])

                self.avgpool = my_model1.avgpool_1a
                
                self.classifier = nn.Linear(1536, self.num_classes)
                
        elif self.NN == 'efficientnet-b4':
                self.feature1 = my_model1
                self.feature2 = my_model2
                
                self.avgpool =  my_model1._avg_pooling
                self.dropout = my_model1._dropout
                self.swish = my_model1._swish
                
                # ori_model = EfficientNet.from_pretrained('efficientnet-b4')
                self.classifier =nn.Linear(1792, num_classes)
                      
        # else:           
        elif self.NN == 'Alexnet':
            self.feature1 = my_model1.features
            self.feature2 = my_model2.features
            
            self.avgpool = my_model1.avgpool
            
            self.classifier = torchvision.models.alexnet(pretrained = False).classifier
            self.classifier[-1] = nn.Linear(4096, self.num_classes)
            
        elif self.NN == 'Vgg16bn':            
            self.feature1 = my_model1.features
            self.feature2 = my_model2.features
            
            self.avgpool = my_model1.avgpool
            
            self.classifier = torchvision.models.vgg16_bn(pretrained = False).classifier
            self.classifier[-1] = nn.Linear(4096, self.num_classes)

        elif self.NN =='Vgg16bn_bn_conv':
            self.feature1 = my_model1.features
            self.feature2 = my_model2.features            
            
            self.classifier = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=256, out_channels = self.num_classes, kernel_size=1),
                                            nn.BatchNorm2d(self.num_classes),
                                            nn.ReLU(inplace=True))      

                                       
    def forward(self,x,y):       

        if self.NN == 'efficientnet-b4':
            x1 = self.feature1.extract_features(x)         
            x2 = self.feature2.extract_features(y)
        else:
            x1 = self.feature1(x)         
            x2 = self.feature2(y)
        
        if self.mtd == 'Conv':
            x3 = torch.cat((x1, x2), dim=1)
            x3 = self.fusion_conv(x3)
                
            if self.NN == 'Alexnet' or self.NN == 'Vgg16bn':
                x3 = self.avgpool(x3)
                x3 = x3.view(x3.size(0), -1)
                x3 = self.classifier(x3)                
            elif self.NN == 'xception':              
                x3 = F.adaptive_avg_pool2d(x3, (1, 1))
                x3 = x3.view(x3.size(0), -1)
                x3 = self.classifier(x3)
            elif self.NN == 'efficientnet-b4':
                x3 = self.dropout(self.avgpool(x3))
                x3 = x3.view(x3.size(0), -1)
                x3 = self.swish(self.classifier(x3))
            elif self.NN == 'Res50' or self.NN == 'inceptionresnetv2':
                x3 = self.avgpool(x3)
                x3 = x3.view(x3.size(0), -1)
                x3 = self.classifier(x3)
            else:
                x3 = self.classifier(x3)
                x3 = F.avg_pool2d(x3, (x3.shape[-2], x3.shape[-1])).squeeze(2).squeeze(2)
        
        elif self.mtd == 'Concat':
            if self.NN == 'Alexnet' or self.NN == 'Vgg16bn':
                x1 = self.avgpool(x1)
                x2 = self.avgpool(x2)
                
                x1 = x1.view(x1.size(0), -1)
                x2 = x2.view(x2.size(0), -1)
                                
            elif self.NN == 'xception':              
                x1 = F.adaptive_avg_pool2d(x1, (1, 1))
                x1 = x.view(x1.size(0), -1)
                
                x2 = F.adaptive_avg_pool2d(x2, (1, 1))
                x2 = x.view(x2.size(0), -1)
                
            elif self.NN == 'inceptionresnetv2':
                x1 = x.view(x1.size(0), -1)
                
                x2 = F.adaptive_avg_pool2d(x2, (8, 8))
                x2 = x.view(x2.size(0), -1)
                
            elif self.NN == 'efficientnet-b4':
                x1 = self.dropout(self.avgpool(x1))
                x1 = x1.view(x1.size(0), -1)
                
                x2 = self.dropout(self.avgpool(x2))
                x2 = x2.view(x2.size(0), -1)

            elif self.NN == 'Res50':
                x1 = x3.view(x1.size(0), -1)
                x2 = x3.view(x2.size(0), -1)
                
            x3 = torch.cat((x1, x2), dim=1)
            
            if self.NN == 'Vgg16bn_bn_conv':
                x3 = self.fusion_conv(x3)
            else:
                x3 = self.fusion_concat(x3)
            if self.NN == 'efficientnet-b4':
                x3 = self.swish(self.classifier(x3))
            else:
                x3 = self.classifier(x3)
            
        return x3