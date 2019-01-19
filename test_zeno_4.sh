#!/bin/bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

export CUDA_VISIBLE_DEVICES=4


# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 10 --nepochs 300 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation mean 2>&1 | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 10 --nepochs 200 --lr 0.1 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 10 --b 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_4.txt



stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 5 --nepochs 200 --lr 0.1 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type label --byz_factor 1 --rho 2 --b 0 --aggregation mean | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_label_mean_8.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.1 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type label --byz_factor 1 --rho 2 --b 0 --aggregation median | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_label_median_8.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 5 --nepochs 200 --lr 0.1 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type label --byz_factor 1 --rho 2 --b 8 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_label_krum_8.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 5 --nepochs 200 --lr 0.1 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type label --byz_factor 1 --rho 20 --b 9 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_label_zeno_8.txt
