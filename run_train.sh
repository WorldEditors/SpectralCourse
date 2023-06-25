#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
nohup python main.py -train_data data/train.raw.dat.npz -test_data data/test.raw.dat.npz -data_type RAW > log.res &
