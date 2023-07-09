# Multi-Scale-Feature-Fusion

![Graphical abstract](https://github.com/endiqq/Multi-Scale-Feature-Fusion/assets/31194584/a018a6f0-ba39-4cbc-9e15-67b0cc50c6ae)


Please cite our work if you feel this post is helpful: https://arxiv.org/abs/2304.12988

Please use below command for training:

python covid_vit_US/bin/main.py --epochs 35 \
                                --dataset data \
                                --action train_mstransfuser_psa \
                                --source ori \
                                --network Res50 \
                                --cos \
                                --root logdir_res50

Please use below command for testing:

python covid_vit_US/bin/main.py --dataset data \
                               --action test_mstransfuser_psa \
                               --source ori \
                               --cos \
                               --root logdir_res50_again\
                               --test_ds 5k
