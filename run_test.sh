#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
# Test Raw Data + Transformer
python main.py -test_data data/test.raw.dat.npz -train 0 -load_model models_trn/checkpoint-5 -data_type RAW > results.dat &

# Test Cumulants + MLP
#python main.py -test_data data/test.cum.dat.npz -train 0 -load_model models_mlp/checkpoint-20 -data_type CUM > results.dat &
