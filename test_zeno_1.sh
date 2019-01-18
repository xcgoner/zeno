#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 10 --nepochs 300 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation mean 2>&1 | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 10 --nepochs 200 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 2 --b 0 --aggregation mean | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10.txt

stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 5 --nepochs 200 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation mean | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_nobyz_mean_0.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation median | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_nobyz_median_0.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 5 --nepochs 200 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 4 --byz_type no_byz --byz_factor 1 --rho 2 --b 4 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_nobyz_krum_0.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 5 --nepochs 200 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 20 --b 4 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_nobyz_zeno_0.txt
