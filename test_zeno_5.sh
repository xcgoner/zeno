#!/bin/bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

export CUDA_VISIBLE_DEVICES=5


# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 200 --b 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_200_16.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 100 --b 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_100_16.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 50 --b 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_050_16.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 25 --b 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_025_16.txt


# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 200 --b 18 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_200_18.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 100 --b 18 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_100_18.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 50 --b 18 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_050_18.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 10 --b 18 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_010_18.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 5 --b 18 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_005_18.txt


stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 200 --b 10 --zeno_size 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_bs_16.txt




