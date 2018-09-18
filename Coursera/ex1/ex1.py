# -*- coding: UTF-8 -*-
import numpy as np


# 初始化数据
def init():
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    data = np.insert(data, 0, np.ones(data.shape[0]), axis=1)
    x = data[:, :2]
    y = data[:, -1]
    y = y.reshape(data.shape[0], 1)
    # 随机初始化系数
    theta = np.array([2, 3])
    theta = theta.reshape(2, 1)
    return x, y, theta


x, y, theta = init()


# 线性回归代价函数
def compute_cost(x, y, theta):
    m = x.shape[0]
    return np.sum((np.dot(x, theta) - y) ** 2) / (2 * m)


print compute_cost(x, y, theta)
