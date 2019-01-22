#!/bin/bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

export CUDA_VISIBLE_DEVICES=2

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 10 --nepochs 300 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 0 --byz_type no_byz --byz_factor 1 --rho 2 --b 0 --aggregation mean 2>&1 | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 10 --nepochs 200 --lr 0.4 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 2 --b 2 --aggregation median | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_2.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 4 --byz_type no_byz --byz_factor 1 --rho 2 --b 4 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_nobyz_krum_0.txt


# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type signflip --byz_factor 1 --rho 200 --b 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type signflip --byz_factor 1 --rho 2 --b 0 --aggregation mean | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_mean_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type signflip --byz_factor 1 --rho 2 --b 0 --aggregation median | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_median_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type signflip --byz_factor 1 --rho 2 --b 8 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_krum_12.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 200 --b 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_label_zeno_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 2 --b 0 --aggregation mean | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_label_mean_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 2 --b 0 --aggregation median | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_label_median_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 2 --b 8 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_label_krum_12.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 12 --byz_type signflip --byz_factor 1 --rho 200 --b 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_signflip_zeno_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 12 --byz_type signflip --byz_factor 1 --rho 2 --b 0 --aggregation mean | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_signflip_mean_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 12 --byz_type signflip --byz_factor 1 --rho 2 --b 0 --aggregation median | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_signflip_median_12.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 12 --byz_type signflip --byz_factor 1 --rho 2 --b 8 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_signflip_krum_12.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 1 --b 10 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_001_10.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 10 --b 8 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_010_08.txt
# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 5 --b 8 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_005_08.txt


# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 400 --zeno_size 8 --b 16 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_label_zeno_12.txt

# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 12 --byz_type label --byz_factor 1 --rho 2 --b 8 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_label_krum_12.txt


# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 1 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 50 --b 14 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_050_14.txt


# stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 3 --nepochs 200 --lr 0.05 --batch_size 50 --iid 0 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 2 --b 8 --aggregation krum | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_noniid_signflip_krum_8.txt

stdbuf -o 0  python mxnet_cnn_cifar10_impl.py --gpu 0 --nrepeats 2 --nepochs 200 --lr 0.05 --batch_size 50 --nworkers 20 --nbyz 8 --byz_type signflip --byz_factor 1 --rho 200 --b 10 --zeno_size 2 --aggregation zeno | tee /home/nfs/cx2/src/byz/zeno/results/test_zeno_cifar10_signflip_zeno_8_bs_2.txt




