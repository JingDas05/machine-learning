# -*- coding: UTF-8 -*-
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *


# eg = []
# eg.append([1, 2, 3])
# eg.append([4, 5, 6])
# print eg

# 构建测试数据集
def loadDataSet(filePath):
    dataMat = []
    labelMat = []
    fr = open(filePath)
    # 遍历测试数据集
    for line in fr.readlines():
        # 每行按照空格拆分
        lineArr = line.strip().split()
        # 取出测试数据集的第一列和第二列构建数据集
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 第三列为标签列
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# dataArr, labelMat = loadDataSet()


# S形函数（阶跃函数）
def sigmoid(inX):
    # exp(-inX) 是取指数，即 e的-inX方
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    # 转换成Numpy的矩阵
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    # 转换成Numpy的矩阵，并且转置
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    # 获取数据矩阵的行，列数
    m, n = shape(dataMatrix)
    # 目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):  # heavy on matrix operations
        # 这里是在计算真实类别和预测类别的差值，接下来就是按照该差值方向调整回归系数
        h = sigmoid(dataMatrix * weights)  # matrix mult
        # 标签标量值 减 阶跃函数值 等于 错误因子
        error = (labelMat - h)  # vector subtraction
        # 重新计算权重因子：将数据集转置，乘错误因子，之后再乘步长？
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights


# dataArr, labelMat = loadDataSet()
# gradAscent(dataArr, labelMat)


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet('testSet.txt')
    dataArr = array(dataMat)
    # 获取数据集行数
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    # 对所有的数据进行分类，将想对应的x,y坐标记录到xcord1 = [],ycord1 = []以及xcord2 = []，ycord2 = []中
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    # 调用figure创建一个绘图对象，包括组成图表的所有元素
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 构建logistic回归方程
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# dataMat, labelMat = loadDataSet()
# weights = gradAscent(dataMat, labelMat)
# plotBestFit(weights.getA())


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # initialize to all ones
    for i in range(m):
        # 此处的h是数值，不是向量
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 随机最优算法 MVP
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        # 构建数据集的索引数组
        dataIndex = range(m)
        # 遍历整个数据集
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            # 随机选取数据集中的数据更新
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 从列表中删除该值
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 遍历训练数据集
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        # 依次遍历每列数据
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 构建当前行的训练数据集
        trainingSet.append(lineArr)
        # 构建标签列
        trainingLabels.append(float(currLine[21]))
    # 求解最优回归系数，迭代次数1000
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    # 用测试数据集计算错误率
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        # 读取测试数据集的每行数据
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 将预测的结果和实际的结果进行比较，如果不相等，错误值+1
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    # 计算错误率
    errorRate = (float(errorCount) / numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate


colicTest()


# 测试10次取平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))
