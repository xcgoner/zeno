#!/bin/bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

export CUDA_VISIBLE_DEVICES=6


stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 200 --b 12 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_200_12.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 100 --b 12 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_100_12.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 50 --b 12 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_050_12.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 25 --b 12 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_025_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 10 --b 12 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_010_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 5 --b 12 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_005_12.txt

stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 200 --b 14 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_200_14.txt
stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 100 --b 14 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_100_14.txt
