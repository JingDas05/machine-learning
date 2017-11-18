# -*- coding: UTF-8 -*-
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator
import Ch03.treePlotter as treePlt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    # dataSet为当前数据集， lables 为按列的顺序的标签
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


# 按照给定特征划分数据集,就是过滤出每行的第axis项是value的行，并且将每行axis项去除
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

# 总体思想就是对于数据的每一列,去掉求熵,比基础数据集的熵减少最多的那列就是特征列
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels 最后一行被当作标签列
    baseEntropy = calcShannonEnt(dataSet)  # 初始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in
                    dataSet]  # create a list of all the examples of this feature 获取dataSet中所有行的第i列，组成数组并且赋值给 featList
        # 去同
        uniqueVals = set(featList)  # get a set of unique values
        # 初始化熵值
        newEntropy = 0.0
        for value in uniqueVals:  # 循环遍历所选列的所有值
            subDataSet = splitDataSet(dataSet, i, value)  # 切分数据集，找到dataSet第i列的值等于value的数据行，去除value,返回过滤后的数据集
            prob = len(subDataSet) / float(len(dataSet))  # 计算 dataSet第i列的值等于value 出现的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算香农熵
        # 计算熵减
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        # 如果熵减小的比以前多,那么这就是最好的划分
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


# myDat, labels = createDataSet()
# print chooseBestFeatureToSplit(myDat)

# 与第2章的classify0部分的投票表决代码类似,找出数组classList中出现次数最多的元素
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    # 字典classCount按照value值降序排序，获取value最大的键值对
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 递归建树，参数为数据集和标签列表
def createTree(dataSet, labels):
    # classList为数据集的所有标签列
    classList = [example[-1] for example in dataSet]
    # 递归停止条件
    # 类别完全相同时，停止划分，算法思想：标签数组中目标元素的数量等于标签数组的长度
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    # 每行元素的数量为1，也即已遍历完所有特征，无法简单的返回唯一的类标签，筛选出出现次数最多的类别作为返回值
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    # 选择特征列
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 特征列的标签
    bestFeatLabel = labels[bestFeat]
    # 初始化根节点
    myTree = {bestFeatLabel: {}}
    # 删除标签数组中的已选特征值
    del (labels[bestFeat])
    # 取出数据集中的特征值列
    featValues = [example[bestFeat] for example in dataSet]
    # 去同
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制标签，所以树不会弄乱已存在的标签，参数是按照引用方式传递的，为了保证每次调用函数createTree()时
        # 不改变原始列表的内容
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        # 去除掉特征列，递归构建树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# myDat, labels = createDataSet()
# myTree = createTree(myDat, labels)
# print myTree


# testVec对应着标签数组中的元素的程序表示，比如featLabels = [‘no surfacing’, 'flippers']
# testVec = [1, 0] 代表着no surfacing出现，flippers不出现的分支
def classify(inputTree, featLabels, testVec):
    # 获取根节点
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    # 变换成程序识别的0或者1等其他与标签对应的值
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 如果是字典，递归调用
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


# myDat, labels = createDataSet()
# mytree = treePlt.retrieveTree(0)
# classify(mytree, labels, [1, 0])

# 存储树，也即序列化树
def storeTree(inputTree, filename):
    # python模块 pickle可以序列化对象
    import pickle
    fw = open(filename, 'w')
    # dump 倾倒，卸下
    pickle.dump(inputTree, fw)
    fw.close()


# 获取树，也即反序列化， grab：抢夺，抢先
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


myTree = treePlt.retrieveTree(0)
# storeTree(myTree, 'classifierStorage1.txt')
# print grabTree('classifierStorage1.txt')

# 隐形眼镜决策树
fr = open('lenses.txt')
# 对于每行数据，根据 \t 分隔字符
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print lensesTree
treePlt.createPlot(lensesTree)
