# -*- coding: UTF-8 -*-
'''
Created on Oct 14, 2010

@author: Peter Harrington
'''
import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 获取叶子节点的数量，确定x轴的长度
def getNumLeafs(myTree):
    numLeafs = 0
    # 获取根节点
    firstStr = myTree.keys()[0]
    # 获取根节点的子节点
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            # 如果是dict递归调用获取树叶节点数量
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取决策树的深度，确定y轴的长度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            # 对于每一个分支都计算深度，+1
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


# 绘制带箭头的注解
# coords 坐标， nodeTxt为注解文本内容，centerPt箭头起始点， parentPt箭头终点（文本处）， nodeType 注解形状
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 在父子节点间填充文本信息 0或者1，也即箭头的中间填写
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    # 在xMid， yMid的位置注释文本txtString
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    # 计算高度和宽度
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    # 根节点
    firstStr = myTree.keys()[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 注释文本
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 画节点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 初始化分支字典
    secondDict = myTree[firstStr]
    # 第二层下移 1.0 / plotTree.totalD
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


# if you do get a dictonary you know it's a tree, and the first element will be another dict
# 主函数，绘制x轴的范围是0.0到1.0，y轴的有效范围是0.0-1.0
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # **的前缀说明这个是字典，参数映射
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    # 初始化方法的全局变量
    plotTree.totalW = float(getNumLeafs(inTree))  # 子节点的总数，树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))  # 树的深度
    plotTree.xOff = -0.5 / plotTree.totalW  # x轴的偏移量
    plotTree.yOff = 1.0  # y轴偏移量
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


myTree = retrieveTree(0)
createPlot(myTree)

# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()

# createPlot()

# myTree = retrieveTree(1)
# print getNumLeafs(myTree)
# print getTreeDepth(myTree)

# createPlot(thisTree)
