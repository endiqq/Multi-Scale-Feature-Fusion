#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:51:54 2020

@author: endiqq
"""

import sys
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
import os
import datetime
from pathlib import Path
import getpass
from torch.nn import DataParallel

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

from model.transfer_models import Classifier
# from model.conv_vit_models import convstem_vit
from model.conv_vit_models_customized import convstem_vit
from model.Fusion_models import Late_Fusion_Net as latefusion
# from model.Fusion_models import Mid_Fusion_Net as middlefusion
# from model.Fusion_models_bce import Late_Fusion_Net as latefusion
from model.fusattention_models import Fus_Attention as  fus_attention
from model.fusattention_models_customized import Fus_Attention as fus_attention_cs
# from model.crossattention_models import Fus_CrossViT as cros_attention
from model.crossattention_models_224 import Fus_CrossViT as cros_attention

from model.ms_transfuser_models_middle import ms_transfuser_mid
from model.ms_transfuser_models_late import ms_transfuser_late
# from model.ms_transfuser_models import ms_transfuser
# from model.ms_transfuser_models_changetransition_layers import ms_transfuser
from model.ms_transfuser_models_res18res34 import ms_transfuser
# from model.ms_transfuser_models_ca import ms_transfuser_ca
# from model.ms_transfuser_models_ca_modified import ms_transfuser_ca
# from model.ms_transfuser_models_pca import ms_transfuser_pca
# from model.ms_transfuser_models_pca_modified import ms_transfuser_pca


from data.utils import Xray_Dataset, Dataloader, Trainer, FusionDataset
from config.config import GlobalConfig

device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train') # train_unlabel
parser.add_argument('--network', default="Res50") #vit_small; vit_base
parser.add_argument('--method', default='Conv')
parser.add_argument('--dataset', default = 'data')

parser.add_argument('--source', default = 'new')
parser.add_argument('--test_ds', default = '5k')
parser.add_argument('--root', default = 'logdir')

parser.add_argument('--warmup-epochs', default=4, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[15, 30, 35], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')

args = parser.parse_args()


def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()


    # hyper-parameters
    K_fold = 5
    # nepoch = 30
    num_classes = [1, 1, 1]
    batch_size = 32
    lr = 1e-3
    img_size = 224

    finetune = ['Entire'] #finetune entire model
    wts = 'Best'

    
    if args.network == 'Res50':
        from model.ms_transfuser_models_pca_res50 import ms_transfuser_pca
        from model.ms_transfuser_models_psa_res50 import ms_transfuser_psa #paralllal self-attention (ours)
        from model.ms_transfuser_models_pfa_res50_changel import ms_transfuser_pfa        
    elif args.network == 'chexpert':
        if img_size == 224:
            from model.ms_transfuser_models_pca import ms_transfuser_pca
            # from model.ms_transfuser_models_pfa import ms_transfuser_pfa
            from model.ms_transfuser_models_psa import ms_transfuser_psa
            # from model.ms_transfuser_models_psa_nopool import ms_transfuser_psa
            # from model.ms_transfuser_models_psa_v import ms_transfuser_psa
        elif img_size == 512:
            from model.ms_transfuser_models_psa_512 import ms_transfuser_psa

    if str(getpass.getuser()) == 'endiqq':
        # STORAGE_ROOT = Path('/home/jby/chexpert_experiments')
        STORAGE_ROOT = Path('covid_vit_US/'+str(args.root))
    else:
        STORAGE_ROOT = Path('/deep/group/aihc-bootcamp-spring2020/cxr_fewer_samples/experiments')
    
    
    # def get_storage_folder(exp_name, exp_type):
    
    #     try:
    #         jobid = os.environ["SLURM_JOB_ID"]
    #     except:
    #         jobid = None
    
    #     datestr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    #     # username = str(getpass.getuser())
    
    #     fname = f'{exp_name}_{exp_type}_{datestr}_SLURM{jobid}' if jobid is not None else f'{exp_name}_{exp_type}_{datestr}'
    
    #     # path_name = STORAGE_ROOT / username / fname
    #     path_name = STORAGE_ROOT / fname
    #     os.makedirs(path_name)
    
    #     print(f'Experiment storage is at {fname}')
    #     return path_name


    


    # define model    
    Trainer_models = Trainer(lr, num_classes, args)
       
    K_Auc=[] #best validation accuracy
    His_Auc=[]#all validation accuracy from all epoechs    
    # train models
    if (args.action == 'train' or args.action == 'train_convvit' or args.action == 'train_fusattention' 
        or args.action == 'train_fusattention_customized'
        or args.action == 'train_crosattention' or args.action == 'train_mstransfuser' 
        or args.action == 'train_mstransfuser_ca' 
        or args.action == 'train_mstransfuser_pca' or args.action == 'train_mstransfuser_pfa'
        or args.action == 'train_mstransfuser_psa'
        or args.action == 'train_mstransfuser_middle' or args.action == 'train_mstransfuser_late'
        or args.action == 'train_latefusion' or args.action == 'train_middlefusion'):
        for ff in finetune:
            print (ff)
            print('-' * 30)   
            for iidx, Dataset in enumerate([args.dataset]):
                print (Dataset)
                print('-' * 30)      
                for k in range(K_fold):            
                    print ('K_fold = %d' % k)
                    folder_name = args.action+'_'+args.network+'_'+Dataset +'_cos_'+str(args.cos)+'_'+str(img_size)+'_'+args.source
                    
                    if args.action == 'train':
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        model = Classifier(num_classes, args.network)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) 
                        
                        dataloaders = Dataloader(Dataset, args).data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                    elif args.action == 'train_convvit':
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        model = convstem_vit(args.network, num_classes)#, embed_dim=256)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        
                        dataloaders = Dataloader(Dataset, args).data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)                    

                    elif args.action == 'train_fusattention':
                        # configuration file
                        config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model = fus_attention(args, config, num_classes, extend=False)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                    elif args.action == 'train_fusattention_customized':
                        # configuration file
                        config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model = fus_attention_cs(args, config, num_classes, extend=False)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)
                        
                    elif args.action == 'train_crosattention':
                        # configuration file
                        # config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model = cros_attention(args.network, multi_scale_enc_depth = 4) #original = 4 , small_depth = 1, large_depth = 1
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                    elif args.action == 'train_mstransfuser':
                        # configuration file
                        config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        # model = ms_transfuser(args, num_classes)
                        # transfuser 
                        model = ms_transfuser(args, config, num_classes)
                        
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                    elif args.action == 'train_mstransfuser_ca':
                        # configuration file
                        # config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model = ms_transfuser_ca(args)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                    elif args.action == 'train_mstransfuser_middle':
                        # configuration file
                        # config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model = ms_transfuser_mid(args, num_classes, 2048)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                    elif args.action == 'train_mstransfuser_late':
                        # configuration file
                        # config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model = ms_transfuser_late(args, num_classes, 2048)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                    elif args.action == 'train_mstransfuser_pca':
                        # configuration file
                        # config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model = ms_transfuser_pca(args, multi_scale_enc_depth = 4)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                    elif args.action == 'train_mstransfuser_pfa':
                        # configuration file
                        config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model = ms_transfuser_pfa(args, config, num_classes, extend=True)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                    elif args.action == 'train_mstransfuser_psa':
                        # configuration file
                        config = GlobalConfig()
                        # tensorboard
                        writer = SummaryWriter(os.path.join(STORAGE_ROOT, folder_name, 'tb_'+args.network+'_'+Dataset +'_'+ str(k)))                     
                        # model, size, pretrained, num_ftrs = tf_learning(num_classes).def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model = ms_transfuser_psa(args, config)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                        dataloaders = Dataloader(Dataset, args).fusion_data_loader(img_size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        hist, best_auc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))#save weights
                        torch.save(Last_model_wts, os.path.join(STORAGE_ROOT, folder_name, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        K_Auc.append(best_auc)
                        His_Auc.append(hist)

                        
                    elif args.action == 'latefusion' or args.action == 'middlefusion':
                        writer = SummaryWriter('runs/'+ args.action+'_'+args.network+'_'+ Dataset +'_'+ str(k))                     
                        
                        model1, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, ff, use_pretrained = True)
                        model2, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model1.load_state_dict(torch.load('Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        # my_model1 = pre_model_conv1#.features                                 
                        model2.load_state_dict(torch.load('Aug_Best_Enh_ijcar_mix_'+ff+'_'+args.network+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))                    
                        
                        if args.action == 'latefusion':
                            model = latefusion(args.network, model1, model2, num_classes)
                        else:
                            embedding_dim = num_ftrs
                            model = middlefusion(args.network, model1, model2, embedding_dim, num_classes, args.method)
                        
                        model_conv = model.to(device)
                        
                        dataloaders = Dataloader(Dataset).fusion_data_loader(size, k, batch_size)
     
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        _, hist, best_acc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, nepoch, params_to_update, writer)
                        
                        if args.action == 'latefusion':
                            torch.save(Best_model_wts, ('CNN_'+args.action+'_Best_lastfc_'+str(num_classes)+'class_'+args.network+'_Sum_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))#save weights
                            torch.save(Last_model_wts, ('CNN_'+args.action+'_Last_lastfc_'+str(num_classes)+'class_'+args.network+'_Sum_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))                                        
                        else:
                            torch.save(Best_model_wts, ('CNN_'+args.action+'_Best_lastconv_'+str(num_classes)+'class_'+args.network+'_'+args.method+'_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))#save weights
                            torch.save(Last_model_wts, ('CNN_'+args.action+'_Last_lastconv_'+str(num_classes)+'class_'+args.network+'_'+args.method+'_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))
                            
                        K_Accuracy.append(best_acc)
                        His_Accuracy.append(hist)
                                  
        # Convert to number from tensor
        K_Auc_num = []
        for i in range(len(K_Auc)):
            K_Auc_num.append(round(float(K_Auc[i]),4))    
        His_Auc_num = []
        for i in range(len(His_Auc)):
            one_NN = []
            for j in range(len(His_Auc[i])):
                one_NN.append(round(float(His_Auc[i][j]),4))
            His_Auc_num.append(one_NN)
                            
        #Save validation accuracy
        file_1 = open(os.path.join(STORAGE_ROOT, folder_name, args.network+'_'+args.action+'_'+Dataset+'_best_val_auc.pickle'),'wb')
        file_2 = open(os.path.join(STORAGE_ROOT, folder_name, args.network+'_'+args.action+'_'+Dataset+'_val_all_auc.pickle'),'wb')
        pickle.dump(K_Auc_num, file_1)
        pickle.dump(His_Auc_num, file_2)
        
    # Test model
    if  (args.action == 'test' or args.action == 'test_convvit' 
         or args.action == 'test_fusattention' or args.action == 'test_fusattention_customized' or args.action == 'test_crosattention' 
         or args.action == 'test_mstransfuser'
         or args.action == 'test_mstransfuser_ca'
         or args.action == 'test_mstransfuser_middle' or args.action == 'test_mstransfuser_late'
         or args.action == 'test_mstransfuser_pca' or args.action == 'test_mstransfuser_pfa'
         or args.action == 'test_mstransfuser_psa'
         or args.action == 'test_latefusion' 
         or args.action == 'test_middlefusion'):

        print ('test')
        
        batch_size = 16
        extend = True
        
        # Container
        all_avg_single_Dataset_acc = []
        all_var_single_Dataset_acc = []
        all_Dataset_acc = []
        
        overall_all_avg_single_Dataset_acc = []
        overall_all_var_single_Dataset_acc = []
        overall_all_Dataset_acc = []
        
        NN_alltype_all_labels = []
        NN_alltype_all_preds = []
        NN_alltype_all_scores = []
        NN_alltype_all_paths = []
        
        for ff in finetune:
            print (ff)
            print('-' * 30)
            
            for iidx, Dataset in enumerate([args.dataset]):
                print (Dataset)
                print('-' * 30)
                
                folder_save = args.action +'_'+ args.network+'_'+Dataset +'_cos_'+\
                    str(args.cos)+'_'+str(img_size)+'_'+args.source+'_'+args.test_ds
                print (folder_save)
                
                if args.action == 'test_latefusion':
                    folder_read_cxr = 'train_'+ args.network+'_'+Dataset +'_cos_'+\
                    str(args.cos)+'_'+str(img_size)+'_'+args.source
                    folder_read_enh = 'train_'+ args.network+'_Train_Mix_cos_'+\
                    str(args.cos)+'_'+str(img_size)+'_'+args.source                    
                    
                    print (folder_read_cxr)
                    print (folder_read_enh)
                else:
                    folder_read = '_'.join(['train']+folder_save.split('_')[1:-1])
                    print (folder_read)
                
                os.makedirs(os.path.join(STORAGE_ROOT, folder_save), exist_ok=True)
                
                if args.action == 'test':
                    model = Classifier(num_classes, args.network)
                    # embedding_dim = num_ftrs
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device)
                elif args.action == 'test_convvit':
                    model = convstem_vit(args.network, num_classes, embed_dim=512) #embed_dim = 256, when it is in logdir_modified
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                elif args.action == 'test_fusattention':
                    config = GlobalConfig()
                    model = fus_attention(args, config, num_classes, extend=extend)
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device) 
                elif args.action == 'test_fusattention_customized':
                    config = GlobalConfig()
                    model = fus_attention_cs(args, config, num_classes, extend=extend)
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device) 
                elif args.action == 'test_crosattention':
                    print ('a')
                    model = cros_attention(args.network, multi_scale_enc_depth = 4) #, small_depth = 2, large_depth = 2
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                elif args.action == 'test_mstransfuser':
                    model = ms_transfuser(args, num_classes)
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                elif args.action == 'test_mstransfuser_middle':
                    model = ms_transfuser_mid(args, num_classes, 2048)
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                elif args.action == 'test_mstransfuser_late':
                    model = ms_transfuser_late(args, num_classes, 2048)
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                elif args.action == 'test_mstransfuser_ca':
                    model = ms_transfuser_ca(args, num_classes)
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                elif args.action == 'test_mstransfuser_pca':
                    model = ms_transfuser_pca(args, multi_scale_enc_depth = 2) #res50 pca = report method depth=2
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device)
                elif args.action == 'test_mstransfuser_pfa':
                    config = GlobalConfig()
                    model = ms_transfuser_pfa(args, config, num_classes, extend=extend)
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device) 
                elif args.action == 'test_mstransfuser_psa':
                    config = GlobalConfig()
                    model = ms_transfuser_psa(args, config)
                    model_conv = DataParallel(model, device_ids=[0,1]).to(device) #DataParallel(model, device_ids=device_ids).to(device) 
                elif args.action == 'test_latefusion':
                    model_conv_1 = Classifier(num_classes, args.network)
                    model_conv_1 = DataParallel(model_conv_1, device_ids=[0,1]).to(device)
                    model_conv_2 = Classifier(num_classes, args.network)
                    model_conv_2 = DataParallel(model_conv_2, device_ids=[0,1]).to(device)
                    # embedding_dim = num_ftrs
                                                        
                single_Dataset_acc = []
                overall_single_Dataset_acc = []
                
                for k in range(K_fold):
                    print (k)
                    if args.source == 'new':    
                        test_ds = os.path.join('Fus-CNNs_COVID-19_US', 'covid_test_'+str(k)+'.txt')
                        # test_ds = os.path.join('create_covid_dataset', 'covid_val_'+str(k)+'.txt')
                    elif args.source == 'ori':
                        if args.test_ds == '5k':
                            test_ds = os.path.join('Fus-CNNs_COVID-19_US', 'test_ds_'+str(k)+'.txt')
                        elif args.test_ds == 'sbu':
                            test_ds = os.path.join('Fus-CNNs_COVID-19_US', 'new_test_dataset2_SBU.txt')
                        elif args.test_ds == 'sbu2':
                            test_ds = os.path.join('Fus-CNNs_COVID-19_US', 'new_test_dataset2_SBU_more.txt')         
                        elif args.test_ds == 'new':
                            test_ds = os.path.join('Fus-CNNs_COVID-19_US', 'new_test_dataset2.txt')
                        # elif args.source == 'ori':
                        #     test_ds = os.path.join('covid_ori_dataset', 'test_dataset2.txt')
                        
                    if args.action == 'test' or args.action == 'test_convvit':    
                        dataloaders = Dataloader(Dataset, args).test_loader(img_size, test_ds, batch_size)
                    else:
                        dataloaders = Dataloader(Dataset, args).test_fusion_loader(img_size, test_ds, batch_size, args.test_ds, img_size)
                                            
                    class_num, all_imgs = Dataloader(Dataset, args).count_imgs(test_ds)
                    print (class_num)
                    
                    # load training weights
                    if args.action == 'test' or args.action == 'test_convvit' \
                        or args.action == 'test_fusattention' or args.action == 'test_fusattention_customized' or args.action == 'test_crosattention'\
                        or args.action == 'test_mstransfuser' or args.action == 'test_mstransfuser_middle' or args.action == 'test_mstransfuser_late'\
                        or args.action == 'test_mstransfuser_pca' or args.action == 'test_mstransfuser_pfa' or args.action == 'test_mstransfuser_psa'\
                        or args.action == 'test_mstransfuser_ca':

                        print (os.path.join(STORAGE_ROOT, folder_read, \
                                                   'Aug_'+ wts+ '_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                       
                        model_conv.load_state_dict(torch.load(os.path.join(STORAGE_ROOT, folder_read, \
                                                   'Aug_'+ wts+ '_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt')))
                        
                        # print (os.path.join(STORAGE_ROOT, folder_read, \
                        #                            'Aug_'+ wts+ '_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt')))
                            
                    elif args.action == 'test_latefusion':    
                        print (os.path.join(STORAGE_ROOT, folder_read_cxr, \
                                                   'Aug_'+ wts+ '_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                       
                        model_conv_1.load_state_dict(torch.load(os.path.join(STORAGE_ROOT, folder_read_cxr, \
                                                   'Aug_'+ wts+ '_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt')))
                            
                        print (os.path.join(STORAGE_ROOT, folder_read_enh, \
                                                   'Aug_'+ wts+ '_Train_Mix_'+ff+'_'+args.network+'_Train_Mix_k_'+str(k+1)+'.pt'))
                       
                        model_conv_2.load_state_dict(torch.load(os.path.join(STORAGE_ROOT, folder_read_enh, \
                                                   'Aug_'+ wts+ '_Train_Mix_'+ff+'_'+args.network+'_Train_Mix_k_'+str(k+1)+'.pt')))
                        

                        model = latefusion(args.network, model_conv_1, model_conv_2, num_classes)
                        # for name, param in  model.named_parameters():
                        #     print (name)
                        model_conv = DataParallel(model, device_ids=[0,1]).to(device)
                        # for name, param in  model_conv.named_parameters():
                        #     print (name)                        
                        
                    elif args.action == 'test_middlefusion':
                        model_conv.load_state_dict(torch.load(('CNN_'+args.action.split('_')[-1]+'_'+wts+'_lastconv_'+str(num_classes)+'class_'+args.network+'_'+args.method+'_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt')))                    
                        
                    # Test accuracy
                    if args.action == 'test' or args.action == 'test_convvit':
                        k_acc, overall_acc, all_preds, all_labels = \
                            Trainer_models.test_model(model_conv, dataloaders, all_imgs, class_num, args.network)
                    else:
                        print ('a')
                        k_acc, overall_acc, all_preds, all_labels = \
                            Trainer_models.test_fusion_1model(model_conv, dataloaders, all_imgs, class_num, args.network)

                    single_Dataset_acc.append(k_acc) # 5 k accuracy
                    overall_single_Dataset_acc.append(overall_acc)
                    
                    # all_labels_list = all_labor args.action == 'test_crosattention'els.tolist()
                    # all_preds_list = all_preds.tolist()
                    # all_scores_list = all_scores.tolist()
                    
                    NN_alltype_all_labels.append(all_labels)
                    NN_alltype_all_preds.append(all_preds)
                    # NN_alltype_all_scores.append(all_scores)
                    # NN_alltype_all_paths.append(all_pat)
    
                temp=[]
                for iii in single_Dataset_acc:
                    temp = temp + iii#[each]
                    normal_acc_5Ks = temp[slice(0,len(temp), len(num_classes))]
                    pneumonia_acc_5Ks = temp[slice(1,len(temp), len(num_classes))]
                    COVID_acc_5Ks = temp[slice(2,len(temp), len(num_classes))] 
                    
                # Calculate mean for each class
                normal_avg_single_Dataset_acc = [round(sum(normal_acc_5Ks)/K_fold,2)]
                pneumonia_avg_single_Dataset_acc = [round(sum(pneumonia_acc_5Ks)/K_fold,2)]
                COVID_avg_single_Dataset_acc = [round(sum(COVID_acc_5Ks)/K_fold,2)]                    
                
                Cross_avg_acc = normal_avg_single_Dataset_acc+pneumonia_avg_single_Dataset_acc+COVID_avg_single_Dataset_acc
                all_avg_single_Dataset_acc.append(Cross_avg_acc)
                
                print(args.network+'_'+'Avg'+'_'+'Acc: normal: %.2f, pneumonia: %.2f, COVID: %.2f' % 
                      (normal_avg_single_Dataset_acc[0], pneumonia_avg_single_Dataset_acc[0], COVID_avg_single_Dataset_acc[0]))
                print('_' * 10)
                
                # #Calcualte var for each class
                # normal_var_single_Dataset_acc = [round(np.std(normal_acc_5Ks),2)]
                # pneumonia_var_single_Dataset_acc = [round(np.std(pneumonia_acc_5Ks),2)]
                # COVID_var_single_Dataset_acc = [round(np.std(COVID_acc_5Ks),2)]                    
               
                # Cross_std_acc = normal_var_single_Dataset_acc+pneumonia_var_single_Dataset_acc+COVID_var_single_Dataset_acc
                # all_var_single_Dataset_acc.append(Cross_std_acc)
                    
                all_Dataset_acc.append(single_Dataset_acc)
                
                # Calculate overall accuracy
                overall_avg_single_Dataset_acc = round(sum(overall_single_Dataset_acc)/K_fold,2)
                overall_all_avg_single_Dataset_acc.append(overall_avg_single_Dataset_acc)
                # # Calculate overall variance
                # overall_var_single_Dataset_acc = round(np.std(overall_single_Dataset_acc),2)
                # overall_all_var_single_Dataset_acc.append(overall_var_single_Dataset_acc)
                # Save all overall accuracy
                overall_all_Dataset_acc.append(overall_single_Dataset_acc)
                # Print result
                print(args.network+'_'+'Average'+'_'+'Acc: %.2f' % (overall_avg_single_Dataset_acc))
                print('_' * 10)
        
        # save accuracy                                   
        # if args.action == 'test' or args.action == 'test_convvit' or args.action == 'test_fusattention'\
        #     or args.action == 'test_crosattention':
        # save variables    
        #import pickle
        file_1 = open(os.path.join(STORAGE_ROOT, folder_save, \
                                   args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_acc.pickle'),'wb') #class accurray
        file_2 = open(os.path.join(STORAGE_ROOT, folder_save, \
                                   args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_avg_single_acc.pickle'),'wb')#
        # file_3 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_var_single_acc.pickle','wb')
        file_4 = open(os.path.join(STORAGE_ROOT, folder_save, \
                                   args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_acc.pickle'),'wb') # overall accuracy
        file_5 = open(os.path.join(STORAGE_ROOT, folder_save, \
                                   args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_avg_single_acc.pickle'),'wb')#
        # file_6 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_var_single_acc.pickle','wb')

        pickle.dump(all_Dataset_acc, file_1) # each class acc each K
        pickle.dump(all_avg_single_Dataset_acc, file_2) # each class avg acc over K
        # pickle.dump(all_var_single_Dataset_acc, file_3) # each class var over K
        pickle.dump(overall_all_Dataset_acc, file_4) # overall each K
        pickle.dump(overall_all_avg_single_Dataset_acc, file_5) # overall avg over K
        # pickle.dump(overall_all_var_single_Dataset_acc, file_6) # overall var over K
        
        #save label and preds for metrics
        file_7 = open(os.path.join(STORAGE_ROOT, folder_save, \
                                   args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_preds.pickle'),'wb')#
        file_8 = open(os.path.join(STORAGE_ROOT, folder_save, \
                                   args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_label.pickle'),'wb')
        pickle.dump(NN_alltype_all_preds, file_7) # preds each K
        pickle.dump(NN_alltype_all_labels, file_8) # label each K
        
        # file_9 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_alltype_scores.pickle','wb')#
        # file_10 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_alltype_paths.pickle','wb')
        # pickle.dump(NN_alltype_all_scores, file_9) # raw prediction score each K
        # pickle.dump(NN_alltype_all_paths, file_10) # image path each K

        # elif args.action=='test_middlefusion':
        #     file_1 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_Enh_ijcar_mix_acc.pickle','wb') #class accurray
        #     file_2 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_avg_single_Enh_ijcar_mix_acc.pickle','wb')#
        #     file_3 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_var_single_Enh_ijcar_mix_acc.pickle','wb')
        #     file_4 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_Enh_ijcar_mix_acc.pickle','wb') # overall accuracy
        #     file_5 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_avg_single_Enh_ijcar_mix_acc.pickle','wb')#
        #     file_6 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_var_single_Enh_ijcar_mix_acc.pickle','wb')

        #     pickle.dump(all_Dataset_acc, file_1) # each class acc each K
        #     pickle.dump(all_avg_single_Dataset_acc, file_2) # each class avg over K
        #     pickle.dump(all_var_single_Dataset_acc, file_3) # each class var over K
        #     pickle.dump(overall_all_Dataset_acc, file_4) # overall each K
        #     pickle.dump(overall_all_avg_single_Dataset_acc, file_5) # overall avg over K
        #     pickle.dump(overall_all_var_single_Dataset_acc, file_6) # overall var over K
        #     #pickle.dump(all_NN_all_Dataset_acc, file_3)
            
        #     #save label and preds for metrics
        #     file_7 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_preds.pickle','wb')#
        #     file_8 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_label.pickle','wb')
        #     pickle.dump(NN_alltype_all_preds, file_7) # preds each K
        #     pickle.dump(NN_alltype_all_labels, file_8) # label each K
            
        #     file_9 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_scores.pickle','wb')#
        #     file_10 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_paths.pickle','wb')
        #     pickle.dump(NN_alltype_all_scores, file_9) # raw predication score each K
        #     pickle.dump(NN_alltype_all_paths, file_10) # image path each K
            
        # else:
        #     file_1 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_Enh_ijcar_mix_acc.pickle','wb') #class accurray
        #     file_2 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_avg_single_Enh_ijcar_mix_acc.pickle','wb')#
        #     file_3 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_var_single_Enh_ijcar_mix_acc.pickle','wb')
        #     file_4 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_Enh_ijcar_mix_acc.pickle','wb') # overall accuracy
        #     file_5 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_avg_single_Enh_ijcar_mix_acc.pickle','wb')#
        #     file_6 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_var_single_Enh_ijcar_mix_acc.pickle','wb')

        #     pickle.dump(all_Dataset_acc, file_1) # each class acc each K
        #     pickle.dump(all_avg_single_Dataset_acc, file_2) # each class avg over K
        #     pickle.dump(all_var_single_Dataset_acc, file_3) # each class var over K
        #     pickle.dump(overall_all_Dataset_acc, file_4) # overall each K
        #     pickle.dump(overall_all_avg_single_Dataset_acc, file_5) # overall avg over K
        #     pickle.dump(overall_all_var_single_Dataset_acc, file_6) # overall var over K
        #     #pickle.dump(all_NN_all_Dataset_acc, file_3)
            
        #     #save label and preds for metrics
        #     file_7 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_preds.pickle','wb')#
        #     file_8 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_label.pickle','wb')
        #     pickle.dump(NN_alltype_all_preds, file_7) # preds each K
        #     pickle.dump(NN_alltype_all_labels, file_8) # label each K
            
        #     file_9 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_scores.pickle','wb')#
        #     file_10 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_paths.pickle','wb')
        #     pickle.dump(NN_alltype_all_scores, file_9) # raw predication score each K
        #     pickle.dump(NN_alltype_all_paths, file_10) # image path each K
            
