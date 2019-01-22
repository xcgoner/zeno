#!/bin/bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

export CUDA_VISIBLE_DEVICES=1

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 10 --nepochs 300 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation mean 2>&1 | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 10 --nepochs 200 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 2 --b 0 --aggregation mean | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 5 --nepochs 200 --lr 0.1 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation mean | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_nobyz_mean_0.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.1 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation median | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_nobyz_median_0.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 5 --nepochs 200 --lr 0.1 --batch_size 50 --nworkers 20 --nbyz 4 --byz_type no_byz --byz_factor 1 --rho 2 --b 4 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_nobyz_krum_0.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 5 --nepochs 200 --lr 0.1 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 20 --b 4 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_nobyz_zeno_0.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation mean | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_nobyz_mean_0.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation median | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_nobyz_median_0.txt


# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 200 --b 10 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_200_10.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 100 --b 10 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_100_10.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 50 --b 10 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_050_10.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 25 --b 10 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_025_10.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 10 --b 10 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_010_10.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 5 --b 10 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_005_10.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 200 --b 8 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_200_08.txt


# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 400 --zeno_size 8 --b 10 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_signflip_zeno_8.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 2 --b 0 --aggregation median | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_signflip_median_8.txt


stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 200 --b 10 --zeno_size 1 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_bs_1.txt








