#!/bin/bash
# Train RAW_DATA + Transformer
#export CUDA_VISIBLE_DEVICES=2
#nohup python main.py -train_data data/train.raw.dat.npz -test_data data/test.raw.dat.npz -data_type RAW -save_dir models_trn > log.train.trn &

# Train Cumulants + MLP
export CUDA_VISIBLE_DEVICES=5
nohup python main.py -train_data data/train.cum.dat.npz -test_data data/test.raw.cum.npz -data_type CUM -save_dir models_mlp > log.train.mlp &
