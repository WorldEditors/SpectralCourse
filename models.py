import sys
import numpy
import matplotlib
from high_order_statistics import cumulants, wavelet_transform
import paddle

import random
import numpy as np
from PIL import Image

import paddle
from paddle import nn
from paddle.nn import Layer, TransformerEncoder, TransformerEncoderLayer, LSTM, Linear, Embedding, Conv2D
import paddle.nn.functional as F
import math

class MLPClassifier(Layer):
    def __init__(self, d_in=4, d_out=24, d_hid1=128, d_hid2=128):
        # 初始化父类中的一些参数
        super(MLPClassifier, self).__init__()
        
        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc_in = Linear(in_features=d_in, out_features=d_hid1)
        self.fc_hid = Linear(in_features=d_hid1, out_features=d_hid2)
        self.fc_out = Linear(in_features=d_hid2, out_features=d_out)
        self.act_hid = nn.ReLU()
        self.act_out = nn.Softmax()
        self.dropout = nn.Dropout(p=0.1)
    
    # 网络的前向计算
    def logits(self, inputs):
        hid1 = self.dropout(self.act_hid(self.fc_in(inputs)))
        hid2 = self.dropout(self.act_hid(self.fc_hid(hid1)))
        return self.fc_out(hid2)

    def forward(self, inputs):
        return self.act_out(self.logits(inputs))
    #
    def loss(self, inputs, labels):
        return F.softmax_with_cross_entropy(self.logits(inputs), labels, soft_label=True)

class LSTMClassifier(Layer):
    def __init__(self, d_in=128, d_hid=128, d_out=24):
        # 初始化父类中的一些参数
        super(LSTMClassifier, self).__init__()
        
        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc_in = Linear(in_features=2, out_features=d_in)
        self.lstm = LSTM(d_in, d_hid, num_layers=2, direction='bidirect')
        self.fc_out = Linear(in_features=d_hid * 2, out_features=d_out)
        self.act_relu = nn.ReLU()
        self.act_out = nn.Softmax()
    
    # 网络的前向计算
    def logits(self, inputs):
        hid1 = self.act_relu(self.fc_in(inputs))
        hid2, (h,c) = self.lstm(hid1)
        return self.fc_out(hid2[:, -1])

    def forward(self, inputs):
        return self.act_out(self.logits(inputs))
    #
    def loss(self, inputs, labels):
        return F.softmax_with_cross_entropy(self.logits(inputs), labels, soft_label=True)

class TransformerClassifier(Layer):
    def __init__(self, d_in=128, d_hid=128, d_out=24):
        # 初始化父类中的一些参数
        super(TransformerClassifier, self).__init__()
        
        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc_in = Linear(in_features=2, out_features=d_in)
        encoder_layer = TransformerEncoderLayer(d_model=d_hid, 
                nhead=8, 
                dim_feedforward=4 * d_hid,
                dropout=0.1,
                activation='gelu',
                normalize_before=False)
        self.transformer = TransformerEncoder(encoder_layer, 4)
        self.fc_out = Linear(in_features=d_hid, out_features=d_out)
        self.act_relu = nn.ReLU()
        self.act_out = nn.Softmax()
    
    # 网络的前向计算
    def logits(self, inputs):
        hid1 = self.act_relu(self.fc_in(inputs))
        hid2 = self.transformer(hid1)
        return self.fc_out(hid2[:, -1])

    def forward(self, inputs):
        return self.act_out(self.logits(inputs))
    #
    def loss(self, inputs, labels):
        return F.softmax_with_cross_entropy(self.logits(inputs), labels, soft_label=True)
