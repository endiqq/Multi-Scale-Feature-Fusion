# Multi-Scale-Feature-Fusion
Please cite our work if you feel this post is helpful: https://arxiv.org/abs/2304.12988

Please use below command for training:

python covid_vit_US/bin/main.py --epochs 35 \
                                --dataset data \
                                --action train_mstransfuser_psa \
                                --source ori \
                                --network Res50 \
                                --cos \
                                --root logdir_res50
