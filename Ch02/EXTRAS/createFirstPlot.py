# -*- coding: UTF-8 -*-
'''
Created on Oct 27, 2010

@author: Peter
'''
from numpy import *
from Ch02 import kNN
import matplotlib
import matplotlib.pyplot as plt

# python的str默认是ascii编码，和unicode编码冲突，如下所示可以解决冲突
import sys
reload(sys)
sys.setdefaultencoding('utf8')

fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = kNN.file2matrix('../data/datingTestSet2.txt')
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# 散点图使用datingDataMat矩阵的第二， 第三列数据,并且添加颜色和大小
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
ax.axis([-2, 25, -0.2, 2.0])
plt.xlabel('每周玩游戏时间百分比')
plt.ylabel('每周消耗冰激凌的升数')
plt.show()
