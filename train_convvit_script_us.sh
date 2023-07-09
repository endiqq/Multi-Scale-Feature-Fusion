#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python covid_vit_US/bin/main.py --epochs 35 \
#                                --dataset data \
#                                --action train_mstransfuser_psa \
#                                --source ori \
#                                --network Res50 \
#                                --cos \
#                                --root logdir_res50

#python covid_vit_US/bin/main.py --epochs 35 \
#                                --dataset data \
#                                --action train_mstransfuser_psa \
#                                --source ori \
#                                --network chexpert \
#                                --cos \
#                                --root logdir_chexpert_pretrained


#python covid_vit/bin/main.py --epochs 35 \
#                             --dataset data \
#                             --action train_crosattention \
#                             --source ori \
#                             --cos \
#                             --root logdir_res50

#python covid_vit/bin/main.py --epochs 35 \
#                             --dataset data \
#                             --action train_fusattention \
#                             --source ori \
#                             --cos \
#                             --root logdir_res50
                                                        
#python covid_vit_US/bin/main.py --epochs 35 \
#                                --dataset data \
#                                --action train_convvit\
#                                --source ori \
#                                --network chexpert_cxr \
#                                --cos \
#                                --root logdir_convvit_cxr
                             
#python covid_vit_US/bin/main.py --epochs 35 \
#                                --dataset Train_Mix \
#                                --action train_convvit\
#                                --source ori \
#                                --network chexpert_enh \
#                                --cos \
#                                --root logdir_convvit_enh
                             
                             
# python covid_vit_US/bin/main.py --epochs 35 \
#                                 --dataset data \
#                                 --action test_convvit\
#                                 --source ori \
#                                 --network chexpert_cxr \
#                                 --cos \
#                                 --root logdir_convvit_cxr
                             
# python covid_vit_US/bin/main.py --epochs 35 \
#                                 --dataset Train_Mix \
#                                 --action test_convvit\
#                                 --source ori \
#                                 --network chexpert_enh \
#                                 --cos \
#                                 --root logdir_convvit_enh

#python covid_vit/bin/main.py  --dataset data \
#                              --action test_crosattention \
#                              --network chexpert \
#                              --source ori \
#                              --cos \
#                              --test_ds sbu
                             
#python covid_vit/bin/main.py --dataset data \
#                             --action test_fusattention \
#                             --network chexpert \
#                             --source ori \
#                             --cos \
#                             --test_ds sbu\
#                             --root logdir_modified

python covid_vit_US/bin/main.py --dataset data \
                               --action test_mstransfuser_psa \
                               --source ori \
                               --cos \
                               --root logdir_res50_again\
                               --test_ds 5k

#python covid_vit_US/bin/main.py --dataset data \
#                                --action test_mstransfuser_psa \
#                                --source ori \
#                                --cos \
#                                --root logdir_chexpert_pretrained\
#                                --test_ds 5k\
#                                --network chexpert

#python covid_vit/bin/main.py --dataset data \
#                             --action test_mstransfuser_middle \
#                             --source ori \
#                             --cos \
#                             --root logdir_res50\
#                             --test_ds sbu

#python covid_vit/bin/main.py --dataset data \
#                             --action test_mstransfuser_late \
#                             --source ori \
#                             --cos \
#                             --root logdir_res50\
#                             --test_ds sbu

#python covid_vit/bin/main.py --dataset data \
#                             --action test_mstransfuser_pca \
#                             --source ori \
#                             --cos \
#                             --root logdir_modified\
#                             --network chexpert\
#                             --test_ds sbu\
