#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:52:08 2020

@author: endiqq
"""



import torch
import torch.nn as nn
import os
from PIL import Image, ImageFile
import time
import torchvision
from torchvision import transforms
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
from sklearn import metrics
import PIL
from barbar import Bar
# from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
             
ImageFile.LOAD_TRUNCATED_IMAGES = True
        
class Xray_Dataset(object):
    def __init__(self, csv_file, dataset, transform=None):
        super(Xray_Dataset, self).__init__()
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.dataset = dataset
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx):   
        cat = self.frame.loc[idx][0].split(' ')
        img_path =  os.path.join(cat[1], self.dataset, cat[2])
        # print (img_path)
        # label = int(cat[3])
        label = np.zeros(3)
        label[int(cat[3])] = 1.0  
        
        if self.dataset == 'data':
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label.astype(np.float32), img_path

    
class FusionDataset(object):
    def __init__(self, csv_file, dataset_1, dataset_2, transform1 = None, transform2 = None):
        super(FusionDataset, self).__init__()
        
        self.frame = pd.read_csv(csv_file, header = None)
        self.transform1 = transform1
        self.transform2 = transform2
        
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
                  
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):        
        cat = self.frame.loc[idx][0].split(' ')
        
        img_path_1 = os.path.join(cat[1], self.dataset_1, cat[2])
        img_path_2 = os.path.join(cat[1], self.dataset_2, cat[2])
        
        img_path = cat[1] + '/' + cat[2]
        # print (img_path)
        
        # label = int(cat[3])
        label = np.zeros(3)
        label[int(cat[3])] = 1.0  
        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2)
        
        # if self.transform is not None:
        img_1 = self.transform1(img_1)
        img_2 = self.transform2(img_2)           
        return img_1, img_2, label.astype(np.float32), img_path
    

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for iidx, content in enumerate(Bar(loader)):
        data, _, path = content
        # print (path)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    
    # del data
    # if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return mean, std

def fusion_online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    channels_sum2, channels_squared_sum2, num_batches2 = 0, 0, 0
    
    for iidx, content in enumerate(Bar(loader)):
        data, data2, _, _ = content
        
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
        
        channels_sum2 += torch.mean(data2, dim=[0, 2, 3])
        channels_squared_sum2 += torch.mean(data2**2, dim=[0,2,3])
        num_batches2 += 1

        # del data, data2
        # if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    
    mean2 = channels_sum2/num_batches2
    std2 = (channels_squared_sum2/num_batches2 - mean2**2)**0.5

    # del data, data2
    # if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return mean, std, mean2, std2

class Dataloader:
    def __init__(self, dataset, args):
        self.Dataset = dataset
        self.args = args
 
    def data_loader(self, size, k, batch_size):
                    
        data_transforms = transforms.Compose([transforms.Resize((size, size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(hue=.05, saturation=.05),
                                              transforms.RandomAffine(10, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                                              transforms.ToTensor()]) 
        
        if self.args.source == 'new':
            dataloaders = {x: torch.utils.data.DataLoader(Xray_Dataset(os.path.join('Fus-CNNs_COVID-19_US', 'covid_'+x+'_'+str(k)+'.txt'),
                                                                       self.Dataset, data_transforms), 
                                                          batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4) for x in ['train','val']}
        elif self.args.source == 'ori':
            dataloaders = {x: torch.utils.data.DataLoader(Xray_Dataset(os.path.join('Fus-CNNs_COVID-19_US', x+'_ds_'+str(k)+'.txt'), 
                                                                    self.Dataset, data_transforms), 
                                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}       


        all_mean = []
        all_std = []
        for iner in ['train', 'val']:
            print (iner)
            mean, std = online_mean_and_sd(dataloaders[iner])
            print (mean, std)
            all_mean.append(mean.numpy())
            all_std.append(std.numpy())
        
        # all_mean = [np.array([0.4758, 0.4758, 0.4758]), np.array([0.4729, 0.4729, 0.4729])]
        # all_std = [np.array([0.2672, 0.2672, 0.2672]), np.array([0.2674, 0.2674, 0.2674])]     
        print (all_mean, all_std)
                        
        data_transforms = {
            'train': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(10, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[0], all_std[0])
                              ]),
            'val': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(10, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[1], all_std[1])
                              ])
            }

        if self.args.source == 'new':
            dataloaders = {x: torch.utils.data.DataLoader(Xray_Dataset(os.path.join('Fus-CNNs_COVID-19_US', 'covid_'+x+'_'+str(k)+'.txt'),
                                                                        self.Dataset, data_transforms[x]), 
                                          batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}
        
        elif self.args.source == 'ori':
            dataloaders = {x: torch.utils.data.DataLoader(Xray_Dataset(os.path.join('Fus-CNNs_COVID-19_US', x+'_ds_'+str(k)+'.txt'), 
                                                                    self.Dataset, data_transforms[x]), 
                                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']} 
        return dataloaders
    
    def fusion_data_loader(self, size, k, batch_size):
                    
        data_transforms = {
            'train': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                              # transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.RandomAffine(14, translate = (0.08,0.08), interpolation=transforms.InterpolationMode.BILINEAR), 
                              transforms.ToTensor(),
                              ]),
            'val': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                              # transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.RandomAffine(14, translate = (0.08,0.08), interpolation=transforms.InterpolationMode.BILINEAR), 
                                                      #resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              ])
            }

        if self.args.source == 'new':
            dataloaders = {x: torch.utils.data.DataLoader(FusionDataset(os.path.join('Fus-CNNs_COVID-19_US', 'covid_'+x+'_'+str(k)+'.txt'), 
                                                                    self.Dataset, 'Train_Mix', data_transforms[x], data_transforms[x]), 
                                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}
        elif self.args.source == 'ori':
            dataloaders = {x: torch.utils.data.DataLoader(FusionDataset(os.path.join('Fus-CNNs_COVID-19_US', x+'_ds_'+str(k)+'.txt'), 
                                                                    self.Dataset, 'Train_Mix', data_transforms[x], data_transforms[x]), 
                                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}        
        # dataloaders2 = 
        all_mean = []
        all_std = []
        for iner in ['train', 'val']:
            print (iner)
            mean1, std1, mean2, std2 = fusion_online_mean_and_sd(dataloaders[iner])
            print (mean1, std1, mean2, std2)
            all_mean.append(mean1.numpy())
            all_std.append(std1.numpy())
            all_mean.append(mean2.numpy())
            all_std.append(std2.numpy())
        
        # if k == 0:
        #     # all_mean = [np.array([0.4746, 0.4746, 0.4746]), np.array([0.2691, 0.2691, 0.2691]),
        #     #             np.array([0.6343, 0.4975, 0.1929]), np.array([0.3631, 0.3229, 0.1086])]
            
        #     # all_std = [np.array([0.4711, 0.4711, 0.4711]), np.array([0.2691, 0.2691, 0.2691]), 
        #     #            np.array([0.6331, 0.4989, 0.1928]), np.array([0.3637, 0.3240, 0.1082])] 
        # elif k==1:
        #     all_mean = [np.array([0.4765, 0.4765, 0.4765]), np.array([0.2736, 0.2736, 0.2736]),
        #                 np.array([0.6351, 0.5003, 0.1967]), np.array([0.3651, 0.3234, 0.1100])]
        #     all_std = [np.array([0.4822, 0.4822, 0.4822]), np.array([0.2730, 0.2730, 0.2730]),
        #                np.array([0.6394, 0.5006, 0.1952]), np.array([0.3643, 0.3233, 0.1081])]
        
        # # ori 
        # if k == 0:
        #     all_mean = [np.array([0.47686774, 0.47686642, 0.476869  ]), np.array([0.6342206 , 0.5024081 , 0.19550087]),
        #                 np.array([0.48124173, 0.48124173, 0.48124173]), np.array([0.6394745, 0.5003375, 0.1967682])]
            
        #     all_std = [np.array([0.27314684, 0.27314624, 0.27314737]), np.array([0.36504588, 0.32476324, 0.10956571]), 
        #                 np.array([0.27421507, 0.27421507, 0.27421507]), np.array([0.3646013 , 0.32372093, 0.10897731])] 
        # elif k==1:
        #     all_mean = [np.array([0.47740364, 0.477403  , 0.4774042 ]), np.array([0.63537216, 0.49889138, 0.1959188 ]),
        #                 np.array([0.47954014, 0.47954014, 0.47954014]), np.array([0.6372742 , 0.49651694, 0.1951055 ])]
        #     all_std = [np.array([0.27322403, 0.27322376, 0.2732243 ]), np.array([0.36568135, 0.32293603, 0.11039178]),
        #                 np.array([0.27317142, 0.27317142, 0.27317142]), np.array([0.36456224, 0.32227936, 0.10811874])]        
        # elif k==2: 
        #     all_mean = [np.array([0.47714493, 0.47714376, 0.47714487]), np.array([0.63494366, 0.5017696 , 0.19607809]), 
        #                 np.array([0.47946015, 0.47946015, 0.47946015]), np.array([0.6373316 , 0.49944678, 0.19587773])] 
        #     all_std = [np.array([0.2731526, 0.2731521, 0.2731527]), np.array([0.3653585 , 0.3245616, 0.11027161]), 
        #                 np.array([0.2729181, 0.2729181, 0.2729181]), np.array([0.3639255 , 0.3236944, 0.10786326])]
        # elif k==3:
        #     all_mean = [np.array([0.47471485, 0.47471428, 0.47471553]), np.array([0.63441914, 0.500961  , 0.19616708]), 
        #                 np.array([0.47980192, 0.47980192, 0.47980192]), np.array([0.63796, 0.49907508, 0.19623448])]
        #     all_std = [np.array([0.2732378 , 0.27323762, 0.27323812]), np.array([0.36529034, 0.3243473 , 0.10980542]), 
        #                 np.array([0.27337623, 0.27337623, 0.27337623]), np.array([0.36408612, 0.3242979 , 0.10892467])]

        # elif k==4:
        #     all_mean = [np.array([0.47471485, 0.47471428, 0.47471553]), np.array([0.63441914, 0.500961  , 0.19616708]), 
        #                 np.array([0.47980192, 0.47980192, 0.47980192]), np.array([0.63796, 0.49907508, 0.19623448])]
        #     all_std = [np.array([0.2732378 , 0.27323762, 0.27323812]), np.array([0.36529034, 0.3243473 , 0.10980542]), 
        #                 np.array([0.27337623, 0.27337623, 0.27337623]), np.array([0.36408612, 0.3242979 , 0.10892467])]

        
        # print (all_mean, all_std)
                        
        data_transforms1 = {
            'train': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              # transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.RandomAffine(14, translate = (0.08,0.08), interpolation=transforms.InterpolationMode.BILINEAR), 
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[0], all_std[0])
                              ]),
            'val': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              # transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.RandomAffine(14, translate = (0.08,0.08), interpolation=transforms.InterpolationMode.BILINEAR), 
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[2], all_std[2])
                              ])
        }
        
        data_transforms2 = {
            'train': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              # transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.RandomAffine(14, translate = (0.08,0.08), interpolation=transforms.InterpolationMode.BILINEAR), 
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[1], all_std[1])
                              ]),
            'val': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              # transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.RandomAffine(14, translate = (0.08,0.08), interpolation=transforms.InterpolationMode.BILINEAR), 
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[3], all_std[3])
                              ])
        }
        
        if self.args.source == 'new':
            dataloaders = {x: torch.utils.data.DataLoader(FusionDataset(os.path.join('Fus-CNNs_COVID-19_US', 'covid_'+x+'_'+str(k)+'.txt'), 
                                                                    self.Dataset, 'Train_Mix', data_transforms1[x], data_transforms2[x]), 
                                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}
        elif self.args.source == 'ori':
            dataloaders = {x: torch.utils.data.DataLoader(FusionDataset(os.path.join('Fus-CNNs_COVID-19_US', x+'_ds_'+str(k)+'.txt'), 
                                                                    self.Dataset, 'Train_Mix', data_transforms1[x], data_transforms2[x]), 
                                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}   

        return dataloaders
    
    def test_loader(self, size, test_file, batch_size):
        data_transforms = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
    
        dataloaders = torch.utils.data.DataLoader(Xray_Dataset(test_file, self.Dataset, data_transforms), batch_size=batch_size, shuffle=True, num_workers=4)
        
        
        if self.Dataset == 'data':
            if self.args.test_ds == 'sbu':
                mean = np.array([0.5122, 0.5122, 0.5122])
                std = np.array([0.2179, 0.2179, 0.2179])
            if self.args.test_ds == 'sbu2':
                mean = np.array([0.5123, 0.5123, 0.5123])
                std = np.array([0.2179, 0.2179, 0.2179])
            # elif self.args.test_ds == 'new':
            #     mean = np.array([0.4967, 0.4967, 0.4967])
            #     std = np.array([0.2439, 0.2439, 0.2439])
            else:
                mean, std = online_mean_and_sd(dataloaders)
                
        elif self.Dataset == 'Train_Mix':
            if self.args.test_ds == 'sbu':
                mean = np.array([0.6643, 0.5312, 0.2094])
                std = np.array([0.3057, 0.2796, 0.0882])
            elif self.args.test_ds == 'sbu2':
                mean = np.array([0.664, 0.5312, 0.2094])
                std = np.array([0.3057, 0.2796, 0.0882]) 
            # elif self.args.test_ds == 'new':
            #     mean = np.array([0.6802, 0.5480, 0.2251])
            #     std = np.array([0.3291, 0.2986, 0.1032])
            else:
                mean, std = online_mean_and_sd(dataloaders)     
        
        # mean, std = online_mean_and_sd(dataloaders)
        # # all_mean = [np.array([0.4758, 0.4758, 0.4758]), np.array([0.4729, 0.4729, 0.4729])]
        # # all_std = [np.array([0.2672, 0.2672, 0.2672]), np.array([0.2674, 0.2674, 0.2674])]  
        # print (mean, std)
        # mean.numpy(), std.numpy()
        
        data_transforms = transforms.Compose([transforms.Resize((size, size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std),
                                              ])

        dataloaders = torch.utils.data.DataLoader(Xray_Dataset(test_file, self.Dataset, data_transforms), batch_size=batch_size, shuffle=True, num_workers=4)
        
        return dataloaders
    
    def test_fusion_loader(self, size, test_file, batch_size, test_ds, img_size):

        data_transforms = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])  
        dataloaders = torch.utils.data.DataLoader(FusionDataset(test_file, self.Dataset, 'Train_Mix', data_transforms, data_transforms), 
                                                  batch_size=batch_size, shuffle=True, num_workers=4)
            

        
        # mean1 = np.array([0.5193, 0.5193, 0.5193]); mean2 = np.array([0.7002, 0.5540, 0.2244])
        # std1 = np.array([0.2499, 0.2499, 0.2499]); std2 = np.array([0.3275, 0.2997, 0.0998])
        if img_size == 224:
            if test_ds == 'sbu':
                mean1 = np.array([0.4942, 0.4942, 0.4942]); mean2 = np.array([0.6781, 0.5450, 0.2060])
                std1 = np.array([0.2363, 0.2363, 0.2363]); std2 = np.array([0.3138, 0.2910, 0.0919])
            elif test_ds == 'new':
                mean1 = np.array([0.4967, 0.4967, 0.4967]); mean2 = np.array([0.6802, 0.5480, 0.2251])
                std1 = np.array([0.2439, 0.2439, 0.2439]); std2 = np.array([0.3291, 0.2986, 0.1032])
            elif self.args.test_ds == 'sbu2':
                mean1 = np.array([0.5123, 0.5123, 0.5123]); mean2 = np.array([0.6644, 0.5312, 0.2094])
                std1 = np.array([0.2178, 0.2178, 0.2178]); std2 = np.array([0.3057, 0.2796, 0.0882]) 
            else:
                mean1, std1, mean2, std2 = fusion_online_mean_and_sd(dataloaders)
        else:
            if test_ds == 'sbu':
                mean1 = np.array([0.5124, 0.5124, 0.5124]); mean2 = np.array([0.6642, 0.5314, 0.2097])
                std1 = np.array([0.2193, 0.2193, 0.2193]); std2 = np.array([0.3200, 0.2898, 0.0916])
            elif test_ds == 'new':
                mean1 = np.array([0.4967, 0.4967, 0.4967]); mean2 = np.array([0.6802, 0.5480, 0.2251])
                std1 = np.array([0.2439, 0.2439, 0.2439]); std2 = np.array([0.3291, 0.2986, 0.1032])
            else:
                mean1, std1, mean2, std2 = fusion_online_mean_and_sd(dataloaders)   
                                                                                             
        # mean1, std1, mean2, std2 = fusion_online_mean_and_sd(dataloaders)
        print (mean1, std1, mean2, std2)
            
        
        data_transforms1 = transforms.Compose([transforms.Resize((size, size)),
              transforms.ToTensor(),
              transforms.Normalize(mean1, std1)
              ])
        
        data_transforms2 = transforms.Compose([transforms.Resize((size, size)),
              transforms.ToTensor(),
              transforms.Normalize(mean2, std2)
              ])

        dataloaders = torch.utils.data.DataLoader(FusionDataset(test_file, self.Dataset, 'Train_Mix', data_transforms1, data_transforms2), 
                                                  batch_size=batch_size, shuffle=True, num_workers=4)
        
        return dataloaders

    
    def count_imgs(self, file):
        class_num = [0, 0, 0]
        frame = pd.read_csv(file, header=None)
        for i in range(len(frame)):
            cat = frame.iloc[i][0].split(' ')
            if cat[3] == '0':
                class_num[0] += 1
            elif cat[3] == '1':
                class_num[1] += 1
            else:
                class_num[-1] += 1
        print (class_num, len(frame))
        return class_num, len(frame)  



def get_loss(output, target, index, device):
    # if cfg.criterion == 'BCE':
    # for num_class in cfg.num_classes:
    #     assert num_class == 1
    target = target[:, index].view(-1)
    # print (target)
    pos_weight = torch.from_numpy(
        np.array([1,1,1], dtype=np.float32)
        ).to(device).type_as(target)
    # if cfg.batch_weight:
    #     if target.sum() == 0:
    #         loss = torch.tensor(0., requires_grad=True).to(device)
    #     else:
    #         weight = (target.size()[0] - target.sum()) / target.sum()
    #         loss = F.binary_cross_entropy_with_logits(
    #             output[index].view(-1), target, pos_weight=weight)
    # else:
    loss = F.binary_cross_entropy_with_logits(
        output[index].view(-1), target, pos_weight=pos_weight[index])

    label = torch.sigmoid(output[index].view(-1)).ge(0.5).float()
    # print (label)
    class_corrects = (target == label).float().sum() #/ len(label)
    # else:
    #     raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return loss, class_corrects



def adjust_learning_rate(optimizer, epoch, args, lr):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if args.cos:
        if epoch < args.warmup_epochs:
            lr_ = lr * epoch / args.warmup_epochs 
        else:
            lr_ = lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    else:  # stepwise lr schedule
        lr_ = lr
        # print (lr_)
        for milestone in args.schedule:
            lr_ *= 0.1 if epoch >= milestone else 1.
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_


class Trainer:
    def __init__(self, lr, num_classes, args):
        # self.ce = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes
        self.args = args
       
    def train_model(self, model, dataloaders, params, writer):
        optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_mean_auc = 0.0
        
        val_auc_history = []
        lr_allep_1per_1iter = []
        
        for epoch in range(self.args.epochs):
                    
            print('Epoch {}/{}'.format(epoch, self.args.epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                iters_per_epoch = len(dataloaders[phase])
                if phase == 'train':
                    model.train()  # Set model to training mode
                    print(optimizer.param_groups[0]['lr'])
                else:
                    model.eval()   # Set model to evaluate mode
                    
                # running_loss = 0.0
                # running_corrects = 0
                
                loss_sum = np.zeros(len(self.num_classes))
                corr_sum = np.zeros(len(self.num_classes))
                
                predlist = list(x for x in range(len(self.num_classes)))
                true_list = list(x for x in range(len(self.num_classes)))
                
                # Iterate over data.
                for index, data in enumerate(Bar(dataloaders[phase])):
                                           
                    if self.args.cos:
                        _lr = adjust_learning_rate(optimizer, epoch + index / iters_per_epoch, self.args, self.lr)
                    else:
                        _lr = adjust_learning_rate(optimizer, epoch, self.args, self.lr)
                    
                    if phase == 'train':
                        lr_allep_1per_1iter.append(_lr)
                        writer.add_scalar('lr', _lr, epoch * len(dataloaders['train']) + index)
                        
                    
                    inputs, labels, img_path = data
                    # labels = torch.from_numpy(np.asarray(labels))
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        output = model(inputs)
                        
                        # loss = self.ce(outputs, labels)
                        # _, preds = torch.max(outputs, 1)
                        
                        # different number of tasks
                        loss = 0
                        for t in range(len(self.num_classes)):
                            loss_t, corr_t = get_loss(output, labels, t, device)
                            loss += loss_t # all losses of 5 classes
                            loss_sum[t] += loss_t.item() # class loss
                            corr_sum[t] += corr_t.item() # class number of corrects
                            
                            # AUC
                            output_tensor = torch.sigmoid(
                                output[t].view(-1)).cpu().detach().numpy()
                            target_tensor = labels[:, t].view(-1).cpu().detach().numpy()
                            if index == 0:
                                predlist[t] = output_tensor
                                true_list[t] = target_tensor
                            else:
                                predlist[t] = np.append(predlist[t], output_tensor)
                                true_list[t] = np.append(true_list[t], target_tensor)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    if phase == 'train':
                        writer.add_scalar('Avg_batch_Loss/train_loss', loss, epoch * len(dataloaders['train']) + index)
                    else:
                        writer.add_scalar('Avg_batch_Loss/val_loss', loss, epoch * len(dataloaders['val']) + index)
                    
                    # # statistics
                    # running_loss += loss.item() * inputs.size(0)
                    # running_corrects += torch.sum(preds == labels.data)
                    
                # if phase == 'train':
                #     scheduler.step()
                #     print (scheduler.get_last_lr()[0])
                #     # scheduler.
                    
                # epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_loss = np.sum(loss_sum) / len(dataloaders[phase].dataset)
                
                if phase == 'train':
                    writer.add_scalar('Epoch_Loss/train_loss', epoch_loss, epoch)
                else:
                    writer.add_scalar('Epoch_Loss/val_loss', epoch_loss, epoch)

                auclist = []
                for i in range(len(self.num_classes)):
                    y_pred = predlist[i]
                    y_true = true_list[i]
                    fpr, tpr, thresholds = metrics.roc_curve(
                        y_true, y_pred, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    auclist.append(auc)

                epoch_mean_auc = np.mean(np.array(auclist))
                
                if phase == 'train':
                    writer.add_scalar('Epoch_Auc/train_auc', epoch_mean_auc, epoch)
                else:
                    writer.add_scalar('Epoch_Auc/val_auc', epoch_mean_auc, epoch)
                
                print('{} Loss: {:.4f} Auc: {:.4f}'.format(phase, epoch_loss, epoch_mean_auc))
                
                # deep copy the model
                if phase == 'val' and epoch_mean_auc > best_mean_auc:
                    best_mean_auc = epoch_mean_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_auc_history.append(epoch_mean_auc)

                # epoch_acc = np.sum(corr_sum).double() / len(dataloaders[phase].dataset)
                
                # # deep copy the model
                # if phase == 'val' and epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(model.state_dict())
                # if phase == 'val':
                #     val_acc_history.append(epoch_acc)

                    
        last_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Auc: {:4f}'.format(best_mean_auc))
    
        return val_auc_history, best_mean_auc, best_model_wts, last_model_wts

    def train_fusion_model(self, model, dataloaders, params, writer):
        optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_mean_auc = 0.0
        
        val_auc_history = []
        lr_allep_1per_1iter = []
        
        for epoch in range(self.args.epochs):
                    
            print('Epoch {}/{}'.format(epoch, self.args.epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                iters_per_epoch = len(dataloaders[phase])           
                if phase == 'train':
                    model.train()  # Set model to training mode
                    print(optimizer.param_groups[0]['lr'])
                else:
                    model.eval()   # Set model to evaluate mode
                    
                # running_loss = 0.0
                # running_corrects = 0
                
                loss_sum = np.zeros(len(self.num_classes))
                corr_sum = np.zeros(len(self.num_classes))
                
                predlist = list(x for x in range(len(self.num_classes)))
                true_list = list(x for x in range(len(self.num_classes)))
                
                # Iterate over data.
                for index, data in enumerate(Bar(dataloaders[phase])):
                    
                    if self.args.cos:
                        _lr = adjust_learning_rate(optimizer, epoch + index / iters_per_epoch, self.args, self.lr)
                    else:
                        _lr = adjust_learning_rate(optimizer, epoch, self.args, self.lr)
                    
                    if phase == 'train':
                        lr_allep_1per_1iter.append(_lr)
                        writer.add_scalar('lr', _lr, epoch * len(dataloaders['train']) + index)
                    
                    inputs_1, inputs_2, labels, img_path = data
                    
                    # online_mean_and_sd(inputs)             
                    # labels = torch.from_numpy(np.asarray(labels))
                    inputs_1 = inputs_1.to(device)
                    inputs_2 = inputs_2.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        output = model(inputs_1, inputs_2)
                        
                        # loss = self.ce(outputs, labels)
                        # _, preds = torch.max(outputs, 1)
                        
                        # different number of tasks
                        loss = 0
                        for t in range(len(self.num_classes)):
                            loss_t, corr_t = get_loss(output, labels, t, device)
                            loss += loss_t # all losses of 5 classes
                            loss_sum[t] += loss_t.item() # class loss
                            corr_sum[t] += corr_t.item() # class number of corrects
                            
                            # AUC
                            output_tensor = torch.sigmoid(
                                output[t].view(-1)).cpu().detach().numpy()
                            target_tensor = labels[:, t].view(-1).cpu().detach().numpy()
                            if index == 0:
                                predlist[t] = output_tensor
                                true_list[t] = target_tensor
                            else:
                                predlist[t] = np.append(predlist[t], output_tensor)
                                true_list[t] = np.append(true_list[t], target_tensor)                       
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()                    
                    
                    if phase == 'train':
                        writer.add_scalar('Avg_batch_Loss/train_loss', loss, epoch * len(dataloaders['train']) + index)
                    else:
                        writer.add_scalar('Avg_batch_Loss/val_loss', loss, epoch * len(dataloaders['val']) + index)
                     
                    # del inputs_1, inputs_2, labels, loss
                    # if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                # epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_loss = np.sum(loss_sum) / len(dataloaders[phase].dataset)
                
                if phase == 'train':
                    writer.add_scalar('Epoch_Loss/train_loss', epoch_loss, epoch)
                else:
                    writer.add_scalar('Epoch_Loss/val_loss', epoch_loss, epoch)
    
                auclist = []
                for i in range(len(self.num_classes)):
                    y_pred = predlist[i]
                    y_true = true_list[i]
                    fpr, tpr, thresholds = metrics.roc_curve(
                        y_true, y_pred, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    auclist.append(auc)
    
                epoch_mean_auc = np.mean(np.array(auclist))
                
                if phase == 'train':
                    writer.add_scalar('Epoch_Auc/train_auc', epoch_mean_auc, epoch)
                else:
                    writer.add_scalar('Epoch_Auc/val_auc', epoch_mean_auc, epoch)
                
                print('{} Loss: {:.4f} Auc: {:.4f}'.format(phase, epoch_loss, epoch_mean_auc))
                
                # deep copy the model
                if phase == 'val' and epoch_mean_auc > best_mean_auc:
                    best_mean_auc = epoch_mean_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_auc_history.append(epoch_mean_auc)
    
                # epoch_acc = np.sum(corr_sum).double() / len(dataloaders[phase].dataset)
                
                # # deep copy the model
                # if phase == 'val' and epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(model.state_dict())
                # if phase == 'val':
                #     val_acc_history.append(epoch_acc)
                
                # del inputs_1, inputs_2, labels, loss
                # if torch.cuda.is_available(): torch.cuda.empty_cache()

                    
        last_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Auc: {:4f}'.format(best_mean_auc))
        
        # del model
        # if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return val_auc_history, best_mean_auc, best_model_wts, last_model_wts
  
    
    def test_model(self, model, dataloaders, dataset_sizes, class_num, network):
          
            #Setup model
            model.eval()
            # General accuracy
            # running_corrects = 0
            
            # # Accuarcy of single class
            # normal_running_corrects = 0
            # pneumonia_running_corrects = 0
            # COVID_running_corrects = 0
        
            # count = 0
            
            corr_sum = np.zeros(len(self.num_classes))
            
            predlist = list(x for x in range(len(self.num_classes)))
            true_list = list(x for x in range(len(self.num_classes)))
            
            # Do test
            for index, data in enumerate(Bar(dataloaders)):
                #get inputs
                img, label, img_path = data        
                img_v = img.to(device)
                label_v = label.to(device)
                # make prediction
                prediction = model(img_v)
                
                # different number of tasks
                for t in range(len(self.num_classes)):
        
                    loss_t, corr_t = get_loss(prediction, label_v, t, device)
                    corr_sum[t] += corr_t.item() # class number of corrects
                    
                    # AUC
                    output_tensor = torch.sigmoid(
                        prediction[t].view(-1)).cpu().detach().numpy()
                    target_tensor = label_v[:, t].view(-1).cpu().detach().numpy()
                    if index == 0:
                        predlist[t] = output_tensor
                        true_list[t] = target_tensor
                    else:
                        predlist[t] = np.append(predlist[t], output_tensor)
                        true_list[t] = np.append(true_list[t], target_tensor)                
                

                   
            auclist = []
            for i in range(len(self.num_classes)):
                y_pred = predlist[i]
                y_true = true_list[i]
                fpr, tpr, thresholds = metrics.roc_curve(
                    y_true, y_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                auclist.append(auc)

            mean_auc = np.mean(np.array(auclist))

            # Class_acc = normal_acc + pneumonia_acc + COVID_acc         
            print(network+'_'+'Single_K_normal_Auc: %.2f, Single_K_pneumonia_Auc: %.2f, Single_K_COVID_Auc: %.2f' %
                  (auclist[0], auclist[1], auclist[2]))
            print(network+'_'+'Overall_Auc: %.2f' % (mean_auc))
            print('-' * 30)


            # Each class accuracy
            if class_num[0] != 0:
                normal_acc = [(float(corr_sum[0]) / len(dataloaders.dataset))*100] # Accuracy for single K
            else:
                normal_acc = [0.]
                
            if class_num[1] != 0:
                pneumonia_acc = [(float(corr_sum[1]) / len(dataloaders.dataset))*100]
            else:
                pneumonia_acc = [0.]
                
            if class_num[2] !=0:
                COVID_acc = [(float(corr_sum[2]) / len(dataloaders.dataset))*100]
                print (float(corr_sum[2]) )
                print (len(dataloaders.dataset))
            else:
                COVID_acc = [0.]
            # Overall accuracy                
            overall_acc = (normal_acc[0]+pneumonia_acc[0]+COVID_acc[0])/3  
                
                
        
            # Combine each class together
            Class_acc = normal_acc + pneumonia_acc + COVID_acc         
            print(network+'_'+'Single_K_normal_Acc: %.2f, Single_K_pneumonia_Acc: %.2f, Single_K_COVID_Acc: %.2f' %
                  (normal_acc[0], pneumonia_acc[0], COVID_acc[0]))
            print(network+'_'+'Overall_Acc: %.2f' % (overall_acc))
            print('-' * 30)
            
            
            return Class_acc, overall_acc, predlist, true_list#, paths, scores
        
    def test_fusion_1model(self, model1, dataloaders, dataset_sizes, class_num, network):

            model1.eval()
            # model2.eval()
            # General accuracy
            # running_corrects = 0
            
            # Accuarcy of single class
            # normal_running_corrects = 0
            # pneumonia_running_corrects = 0
            # COVID_running_corrects = 0
            # Pedicle_running_corrects = 0
        
            # count = 0
            
            corr_sum = np.zeros(len(self.num_classes))
            
            predlist = list(x for x in range(len(self.num_classes)))
            true_list = list(x for x in range(len(self.num_classes)))            
            
            # Do test
            for index, data in enumerate(Bar(dataloaders)):
                #get inputs
                img_1, img_2, label, img_path = data
                # print (label)
                img_1_v = img_1.to(device)
                img_2_v = img_2.to(device)
                label_v = label.to(device)
                # make prediction
                prediction = model1(img_1_v, img_2_v)
                # print (prediction)
                # prediction2 = model2(img_2_v)
        #        prediction = model_conv(img_v) # Used for test, comment when in real
        
                # different number of tasks
                for t in range(len(self.num_classes)):
        
                    loss_t, corr_t = get_loss(prediction, label_v, t, device)
                    corr_sum[t] += corr_t.item() # class number of corrects
                    
                    # AUC
                    output_tensor = torch.sigmoid(
                        prediction[t].view(-1)).cpu().detach().numpy()
                    target_tensor = label_v[:, t].view(-1).cpu().detach().numpy()
                    if index == 0:
                        predlist[t] = output_tensor
                        true_list[t] = target_tensor
                    else:
                        predlist[t] = np.append(predlist[t], output_tensor)
                        true_list[t] = np.append(true_list[t], target_tensor)                
                

            auclist = []
            for i in range(len(self.num_classes)):
                y_pred = predlist[i]
                y_true = true_list[i]
                fpr, tpr, thresholds = metrics.roc_curve(
                    y_true, y_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                auclist.append(auc)

            mean_auc = np.mean(np.array(auclist))

            # Class_acc = normal_acc + pneumonia_acc + COVID_acc         
            print(network+'_'+'Single_K_normal_Auc: %.2f, Single_K_pneumonia_Auc: %.2f, Single_K_COVID_Auc: %.2f' %
                  (auclist[0], auclist[1], auclist[2]))
            print(network+'_'+'Overall_Auc: %.2f' % (mean_auc))
            print('-' * 30)
   
            # Each class accuracy
            if class_num[0] != 0:
                normal_acc = [(float(corr_sum[0]) / len(dataloaders.dataset))*100] # Accuracy for single K
                # print (float(corr_sum[0]) )
            else:
                normal_acc = [0.]
                
            if class_num[1] != 0:
                pneumonia_acc = [(float(corr_sum[1]) / len(dataloaders.dataset))*100]
                # print (float(corr_sum[1]) )
            else:
                pneumonia_acc = [0.]
                
            if class_num[2] !=0:
                COVID_acc = [(float(corr_sum[2]) / len(dataloaders.dataset))*100]
                # print (float(corr_sum[2]) )
                # print (len(dataloaders.dataset))
            else:
                COVID_acc = [0.]
            # Overall accuracy                
            overall_acc = (normal_acc[0]+pneumonia_acc[0]+COVID_acc[0])/3  
                
                 
            # Combine each class together
            Class_acc = normal_acc + pneumonia_acc + COVID_acc         
            print(network+'_'+'Single_K_normal_Acc: %.2f, Single_K_pneumonia_Acc: %.2f, Single_K_COVID_Acc: %.2f' %
                  (normal_acc[0], pneumonia_acc[0], COVID_acc[0]))
            print(network+'_'+'Overall_Acc: %.2f' % (overall_acc))
            print('-' * 30)
            
            
            return Class_acc, overall_acc, predlist, true_list
        
        
        
                    # # statistics
                    # running_loss += loss.item() * inputs_1.size(0)
                    # running_corrects += torch.sum(preds == labels.data)
                
                # for param_group in optimizer.param_groups:
                #     print(param_group['lr'])
                    
                # if phase == 'train':
                #     scheduler.step()
                #     print (scheduler.get_last_lr()[0])
                #     # scheduler.
                    
        #         epoch_loss = running_loss / len(dataloaders[phase].dataset)
                
        #         if phase == 'train':
        #             writer.add_scalar('Epoch_Loss/train_loss', epoch_loss, epoch)
        #         else:
        #             writer.add_scalar('Epoch_Loss/val_loss', epoch_loss, epoch)
                    
        #         epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
        #         if phase == 'train':
        #             writer.add_scalar('Epoch_Acc/train_acc', epoch_acc, epoch)
        #         else:
        #             writer.add_scalar('Epoch_Acc/val_acc', epoch_acc, epoch)
                
        #         print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
        #         # deep copy the model
        #         if phase == 'val' and epoch_acc > best_acc:
        #             best_acc = epoch_acc
        #             best_model_wts = copy.deepcopy(model.state_dict())
        #         if phase == 'val':
        #             val_acc_history.append(epoch_acc)
                    
        #         all_epochs_loss.append(epoch_loss)
                    
        # last_model_wts = copy.deepcopy(model.state_dict())
        # time_elapsed = time.time() - since
        # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))
        
        # return model, val_acc_history, best_acc, best_model_wts, last_model_wts#, all_epochs_loss
        
        
                        
                # _, preds = torch.max(prediction.data, 1)
                
                # # statistics
                # running_corrects += torch.sum(preds == label_v)
                # # from GPU to CPU
                # preds = preds.cpu()
                # label = label.cpu()
                
                # if index == 0:
                #     labell = label
                #     predd = preds
                #     scores = prediction.data
                #     paths = list(img_path)
                    
                # else:
                #     labell = torch.cat((labell, label), dim=0)
                #     predd = torch.cat((predd, preds), dim=0)
                #     scores = torch.cat((scores, prediction.data), dim=0)
                #     paths = paths+list(img_path)
                   
                # # Calculate Class accuracy
                # for i in range(len(preds)):
                #     if preds.numpy()[i] == label.numpy()[i]:
                #         if preds.numpy()[i] == 0:
                #             normal_running_corrects = normal_running_corrects+1
                #         elif preds.numpy()[i] == 1:
                #             pneumonia_running_corrects = pneumonia_running_corrects+1
                #         elif preds.numpy()[i] == 2:
                #             COVID_running_corrects = COVID_running_corrects+1
                               
                # # count how many images
                # count = count + 1