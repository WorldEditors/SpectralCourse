import sys
import numpy
import matplotlib
import paddle
import argparse

import random
import numpy as np
from PIL import Image

import paddle
from paddle import nn
from paddle.nn import Layer, Linear, Embedding, Conv2D
from models import MLPClassifier, LSTMClassifier, TransformerClassifier
from high_order_statistics import cumulants, wavelet_transform
from data_processing import DataLoader
import paddle.nn.functional as F
import math

def load_model(model, opt, file_name, load_opt=False):
    layer_state_dict = paddle.load("%s.pdparams"%file_name)
    model.set_state_dict(layer_state_dict)
    if(load_opt):
        opt_state_dict = paddle.load("%s.pdopt"%file_name)
        opt.set_state_dict(opt_state_dict)

def save_model(model, opt, file_prefix, epoch):
    save_file_name = "%s-%d"%(file_prefix, epoch)
    print("Saving models to %s"%(save_file_name))
    paddle.save(model.state_dict(), save_file_name + ".pdparams")
    paddle.save(opt.state_dict(), save_file_name + ".pdopt")

def train_model(dataset, model, 
    batch_size=32,
    save_file=None,
    load_file=None,
    max_epoch=20,
    ):
    lr = paddle.optimizer.lr.ExponentialDecay(learning_rate=1.0e-4, gamma=0.90)
    opt = paddle.optimizer.AdamW(lr, parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
            )
    #print(model.parameters())
    model.train()
    stat_freq = 1.0
    save_freq = 5
    epoch = 0
    while epoch < max_epoch:
        cur_ratio = stat_freq
        acc_loss = []
        epoch_loss = []
        epoch += 1
        print("Training epoch %d start ..." % epoch)
        for r,X,Y,snr in dataset.get_batch(batch_size):
            loss = model.loss(paddle.to_tensor(X.astype("float32")), paddle.to_tensor(Y.astype("float32")))
            loss.backward()
            opt.step()
            opt.clear_grad()
            acc_loss.append(loss.numpy())
            epoch_loss.append(loss.numpy())
            
            if(100.0 * r > cur_ratio):
                print("Finished data percentage: %.1f, average training loss: %.3f, learning rate: %.5f" % (100.0 * r, numpy.mean(acc_loss), opt.get_lr()))
                cur_ratio += stat_freq
                acc_loss = []
        print("Training epoch %d finished, Epoch average loss %f" % (epoch, numpy.mean(epoch_loss)))
        lr.step()
        if(epoch % save_freq == 0):
            save_model(model, opt, "./models/checkpoint", epoch)

def evaluation(dataset, model,
    load_model_file,
    batch_size=16
    ):
    prec = []
    prec_key = dict()
    snrs = []
    for r,X,Y,snr in dataset.get_batch(batch_size):
        model.eval()
        pred = model.forward(paddle.to_tensor(X.astype("float32")))
        cur_prec = numpy.sum(pred.numpy() * Y, axis=-1)
        prec.append(cur_prec)
        for i, snr_key in enumerate(snr):
            snr_t_key = int(snr_key)
            if(snr_t_key not in prec_key):
                prec_key[snr_t_key] = []
            else:
                prec_key[snr_t_key].append(cur_prec[i])
        snrs.append(numpy.squeeze(snr))
    prec = numpy.concatenate(prec, axis=0)
    for key in prec_key:
        print("SNR Evaluation:\t%d\t%f" % (key, numpy.mean(prec_key[key])))
    print("Evaluation Results:", numpy.mean(prec))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Preprocessing the training and testing data')

    parser.add_argument('-train_data', type=str)
    parser.add_argument('-test_data', type=str)
    parser.add_argument('-data_type', type=str, default='RAW')
    parser.add_argument('-train', type=int, default=1)
    parser.add_argument('-load_model', type=str)

    args = parser.parse_args()
    file_type = args.data_type
    if(file_type == 'CUM'):
        model = MLPClassifier()
    elif(file_type == 'RAW'):
        model = TransformerClassifier()

    if(args.train):
        assert args.train_data is not None
        train_dataset = DataLoader(args.train_data, file_type='NPZ')
        train_model(train_dataset, model)
    else:
        assert args.test_data is not None
        if(args.load_model is not None):
            load_model(model, None, args.load_model, load_opt=False)
        test_dataset = DataLoader(args.test_data, file_type='NPZ')
        evaluation(test_dataset, model, args.load_model)
