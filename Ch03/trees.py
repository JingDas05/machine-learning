# -*- coding: UTF-8 -*-
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # the the number of unique elements and their occurance
    # 记录各个元素的数量以及出现次数
    for featVec in dataSet:
        # 获取数据的最后一列作为标签
        currentLabel = featVec[-1]
        # 如果labelCounts字典还没有currentLabel就初始化为0，否则数量+1
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 遍历 便签：数量 数据集
    for key in labelCounts:
        # 获取数量转化为float，并且计算概率
        prob = float(labelCounts[key]) / numEntries
        # 计算香农熵
        shannonEnt -= prob * log(prob, 2)  # log base 2
    # 返回香农熵
    return shannonEnt


# 按照给定特征划分数据集,就是找到每行的 第 axis 个数，之后将第 axis 个数去掉，同时将本行数据添加到集合中返回
# dataSet 待划分的数据集
# axis 划分数据集的特征
# value 特征的返回值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # chop out axis used for splitting
            # 去除特征值
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    # 返回新数组
    return retDataSet


# myDat, labels = createDataSet()
# print myDat
# print labels
# 找到第0项是1的并且把这项去掉
# print splitDataSet(myDat, 0, 1)

# 总体思想就是对于数据的每一列,出去掉求熵,比基础数据集的熵减少最多的那列就是特征列
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels 最后一行被当作标签列
    baseEntropy = calcShannonEnt(dataSet) # 初始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet] # create a list of all the examples of this feature 获取dataSet中所有行的第i列，组成数组并且赋值给 featList
        # 去同
        uniqueVals = set(featList)  # get a set of unique values
        # 初始化熵值
        newEntropy = 0.0
        for value in uniqueVals:  # 循环遍历所选列的所有值
            subDataSet = splitDataSet(dataSet, i, value)  # 切分数据集，找到dataSet第i列的值等于value，去除value,返回数据集
            prob = len(subDataSet) / float(len(dataSet))  # 计算 dataSet第i列的值等于value 出现的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算香农熵
        # 计算熵减
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        # 如果熵减小的比以前多,那么这就是最好的划分
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


myDat, labels = createDataSet()
print chooseBestFeatureToSplit(myDat)


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
