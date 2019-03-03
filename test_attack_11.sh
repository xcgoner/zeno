#!/bin/bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

export CUDA_VISIBLE_DEVICES=0,1,2,3



# stdbuf -o 0 python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 25 --nbyz 11 --byz_type signflip --byz_factor 1 --byz_start 100 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_bs_1.txt

gpu=2
aggr="krum"
nrepeats=2

logdir=/home/nfs/cx2/src/byz_attack/results

logfile=$logdir/byz_attack_${aggr}_baseline.log

> $logfile

stdbuf -o 0 python -u mxnet_cnn_cifar10_impl.py --gpu ${gpu} --nrepeats ${nrepeats} --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 25 --nbyz 0 --byz_type no_byz --byz_factor 1 --byz_start 1000 --aggregation ${aggr} | tee ${logfile}







