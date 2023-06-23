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
from paddle.nn import Layer, Linear, Embedding, Conv2D
import paddle.nn.functional as F
import math

class Classifier(Layer):
    def __init__(self, d_in, d_out):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()
        
        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc_in = Linear(in_features=d_in, out_features=d_hid1)
        self.fc_hid = Linear(in_features=d_hid1, out_features=d_hid2)
        self.fc_out = Linear(in_features=d_hid2, out_features=d_out)
        self.act_hid = nn.ReLU()
        self.act_out = nn.Softmax()
    
    # 网络的前向计算
    def logits(self, inputs):
        hid1 = self.act_hid(self.fc_in(inputs))
        hid2 = self.act_hid(self.fc_hid(hid1))
        return self.fc_hid(hid2)

    def forward(self, inputs):
        return self.act_out(self.logits(inputs))

    #
    def loss(self, inputs, labels):
        return F.softmax_with_cross_entropy(self.logits(inputs), labels)
    
    def save(self, file_name):
        
