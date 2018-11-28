# -*- coding: UTF-8 -*-
import scipy.io as scio
import random
import numpy as np

# 这个为训练数据集
data = scio.loadmat('ex4data1.mat')
theta = scio.loadmat('ex4weights.mat')

# print type(data['X'])
# print type(data['y'])

# x_index = np.arange(0, data['y'].shape[0])
# 随机打乱x的顺序
# random.shuffle(x_index)
# 获取前100个值
# X = data['X'][x_index[0:100]]

X = data['X']
y = data['y']
Theta1 = theta['Theta1']
Theta2 = theta['Theta2']


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_param):
    Theta1