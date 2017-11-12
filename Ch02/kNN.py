# -*- coding: UTF-8 -*-
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt


# inX 用于分类的向量
# dataSet 输入的训练样本集
# labels 标签向量
# k 选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    # 获取dataSet行数
    dataSetSize = dataSet.shape[0]
    # 调用tile函数,生成目标矩阵，之后与数据集相减
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 平方
    sqDiffMat = diffMat ** 2
    # axis=0 是按行求和， axis=1是按列求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # 对于数组排序，并且按照升序返回相对应的index
    sortedDistIndicies = distances.argsort()
    # 定义字典
    classCount = {}
    # range(k) 生成 0, 1, 2，对前k名的标签进行计数，存储到classCount中，key是label,value是出现的个数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 如果存在取值加1，如果不存在，取默认值之后再加1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 调用sorted函数，对字典进行排序，排序的字段是value(引用了操作符模块的功能)，默认是升序，这里reverse=True，为降序
    # 根据出现的次数降序，不改变原数组
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的标签
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# group, labels = createDataSet()
# print(classify0([0, 0], group, labels, 3))

# 构建约会网站的测试样本
def file2matrix(filename):
    fr = open(filename)
    # 获取文件的行数
    numberOfLines = len(fr.readlines())
    # 准备返回的矩阵
    returnMat = zeros((numberOfLines, 3))
    # 准备返回标签
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # 去除头尾 \t字符
        line = line.strip()
        # 根据 \t 分隔字符，返回数组
        listFromLine = line.split('\t')
        # 将 listFromLine[0:3] 追加到 returnMat 中，索引是index
        returnMat[index, :] = listFromLine[0:3]
        # 将listFromLine的最后一列转化成int型追加到classLabelVector中，构建标签列
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')


# print datingDataMat
# print datingLabels

# 自动将数字特征值转化为0到1的区间，计算公式为 newValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
    # 选取列的最小值
    minVals = dataSet.min(0)
    # 选取列的最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # 构建零矩阵，行列数与dataSet相同
    normDataSet = zeros(shape(dataSet))
    # 获取矩阵行数
    m = dataSet.shape[0]
    # 计算 oldValue-min
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 计算 (oldValue-min)/(max-min)
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals, maxVals


# norMat, ranges, minVals, maxVals = autoNorm(datingDataMat)
# print norMat
# print ranges
# print minVals
# print maxVals


def datingClassTest():
    hoRatio = 0.10  # hold out 10%
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')
    # 数值归一化
    normMat, ranges, minVals, maxVals = autoNorm(datingDataMat)
    # 获取数据集行数
    m = normMat.shape[0]
    # 测试的行数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print
        "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        # 如果判断的标签和数据集里面的不一致,就将 errorCount 加1
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print
    "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print
    errorCount


# datingClassTest()

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    # 循环读取32行
    for i in range(32):
        lineStr = fr.readline()
        # 对于每一行的数据添加到 1x1024矩阵中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # 加载训练数据集
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # 循环读取所有的数据集
    for i in range(m):
        # 获取文件名
        fileNameStr = trainingFileList[i]
        # 去除掉 .txt
        fileStr = fileNameStr.split('.')[0]
        # 获取实际的数字值
        classNumStr = int(fileStr.split('_')[0])
        # 追加数字
        hwLabels.append(classNumStr)
        # 赋值,trainingMat的第i行的所有列等于1x1024的矩阵
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    # 循环迭代 获取测试数据集
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        # 去除掉 .txt 后缀
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        # 转化成1x1024向量,作为查询参数
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        # knn算法,获取数字
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print
        "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        # 如果knn计算出来的数字不等于实际的数字,errorCount 加1
        if (classifierResult != classNumStr): errorCount += 1.0
    print
    "\nthe total number of errors is: %d" % errorCount
    print
    "\nthe total error rate is: %f" % (errorCount / float(mTest))


handwritingClassTest()
