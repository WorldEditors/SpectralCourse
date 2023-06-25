#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
python main.py -test_data data/test.raw.dat.npz -train 0 -load_model models/checkpoint-5 -data_type RAW > res_trn.dat
