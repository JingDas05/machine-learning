# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def init():
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    # 取数据集前两列，并且添加偏置项 列1
    # insert 的逻辑需要注意下data[:, :2]是100×2，np.ones((1, data.shape[0]))是1×100
    # 可以理解为对所有的axis=1(行操作)的数据在0列之前添加 np.ones
    x = np.insert(data[:, :2], 0, np.ones((1, data.shape[0])), axis=1)
    y = data[:, -1].reshape(data.shape[0], 1)
    theta = np.zeros(x.shape[1])
    theta = np.array([-24, 0.2, 0.2])
    return x, y, theta


def compute_cost(theta, x, y):
    m = x.shape[0]
    n = x.shape[1]
    # 当进行矩阵运算的时候，记得要重新结构化矩阵 不要出现（3L，）这样的shape
    temporary_theta = theta.reshape((n, 1))
    j = float(1) / m * np.sum(-y * np.log(sigmoid(np.dot(x, temporary_theta))) - (np.ones((m, 1)) - y) * np.log(
        np.ones((m, 1)) - sigmoid(np.dot(x, temporary_theta))))
    return j


def gradient_descent(theta, x, y):
    m = x.shape[0]
    n = x.shape[1]
    # 当进行矩阵运算的时候，记得要重新结构化矩阵 不要出现（3L，）这样的shape
    temporary_theta = theta.reshape((n, 1))
    theta = float(1) / m * np.sum((sigmoid(np.dot(x, temporary_theta)) - y) * x, axis=0)
    return theta


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(theta, x):
    return np.round(sigmoid(x * theta))


x, y, theta = init()
# j = compute_cost(theta, x, y)
# theta = gradient_descent(theta, x, y)
# print j
# print theta.shape

result = op.minimize(fun=compute_cost, x0=theta, args=(x, y), method='BFGS', jac=gradient_descent)


def caculate_accuracy(result, x, y):
    m = x.shape[0]
    n = x.shape[1]
    trained_theta = np.array(result.x).reshape((n, 1))
    print predict(trained_theta, x)


print(np.array(result.x).shape)
