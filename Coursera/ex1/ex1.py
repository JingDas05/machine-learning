# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 初始化数据
def init():
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    # 在第一列插入 偏置项
    data = np.insert(data, 0, np.ones(data.shape[0]), axis=1)
    x = data[:, :2]
    y = data[:, -1].reshape(data.shape[0], 1)
    # 随机初始化系数
    theta = np.zeros([2, 1])
    return x, y, theta


# 计算代价函数
def compute_cost(x, y, theta):
    """
    x 数据
    y 标签
    theta 系数
    """
    m = x.shape[0]
    return np.sum((np.dot(x, theta) - y) ** 2) / (2 * m)


# 梯度下降
def gradient_descent(x, y, theta, alpha, num_iters):
    j_history = np.zeros((num_iters, 1))
    """
    x 数据 : m × n
    y 标签 : m × 1
    theta 初始化系数 n × 1
    alpha 学习速率
    num_iters 迭代次数
    """
    for each in np.arange(num_iters):
        m = x.shape[0]
        n = x.shape[1]
        temporary_theta = theta
        # np.dot(x, theta)-y : m × 1
        # theta : 1 × n
        # 核心方法，向量化操作，首先根据theta计算实际y值，之后计算数据集里面各个特征值和计算误差的点乘，
        # 之后各个特征值求和
        theta = theta - alpha / m * np.sum((np.dot(x, temporary_theta) - y) * x, axis=0).reshape(n, 1)
        j_history[each] = compute_cost(x, y, theta)
    return theta, j_history


x, y, theta = init()

theta, j_history = gradient_descent(x, y, theta, 0.01, 1500)

x_data = x[:, 1]
plt.figure(1)
plt.scatter(x_data, y, marker='x')

x = np.arange(start=np.min(x_data), stop=np.max(x_data), step=0.1)
y = theta[0] + x * theta[1]

plt.plot(x, y)
plt.show()
