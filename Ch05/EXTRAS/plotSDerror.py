# -*- coding: UTF-8 -*-
'''
Created on Oct 6, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Ch05 import logRegres


def stocGradAscent0(dataMatrix, classLabels):
    # 获取矩阵的维数 m × n
    m, n = shape(dataMatrix)
    alpha = 0.5
    # 生成 n × 1 矩阵
    weights = ones(n)  # initialize to all ones
    weightsHistory = zeros((500 * m, n))
    for j in range(500):
        for i in range(m):
            h = logRegres.sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
            weightsHistory[j * m + i, :] = weights
    return weightsHistory


def stocGradAscent1(dataMatrix, classLabels):
    # 获取矩阵的维数 m × n,m 等于100
    m, n = shape(dataMatrix)
    alpha = 0.4
    # 生成 n × 1 矩阵
    weights = ones(n)  # initialize to all ones
    weightsHistory = zeros((40 * m, n))
    for j in range(40):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选取一条数据记录
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = logRegres.sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            # print error
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 记录weights，注意这个地方的 j * m + i ，j为迭代的次数，m为数据总量100
            weightsHistory[j * m + i, :] = weights
            del (dataIndex[randIndex])
    print weights
    return weightsHistory


dataMat, labelMat = logRegres.loadDataSet('../testSet.txt')
dataArr = array(dataMat)
myHist = stocGradAscent1(dataArr, labelMat)

n = shape(dataArr)[0]  # number of points to create
xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []

markers = []
colors = []

fig = plt.figure()
ax = fig.add_subplot(311)
type1 = ax.plot(myHist[:, 0])
plt.ylabel('X0')
ax = fig.add_subplot(312)
type1 = ax.plot(myHist[:, 1])
plt.ylabel('X1')
ax = fig.add_subplot(313)
type1 = ax.plot(myHist[:, 2])
plt.xlabel('iteration')
plt.ylabel('X2')
plt.show()
