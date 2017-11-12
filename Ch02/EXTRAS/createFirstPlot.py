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
datingDataMat, datingLabels = kNN.file2matrix('../data/datingTestSet2.txt')

# 创建一幅图
fig = plt.figure()
# subplot()命令会指定一个坐标系，默认是subplot(111)
# 111参数分别说明行的数目 numrows，列的数目 numcols，第几个图像fignum（fignum的范围从1到numrows*numcols）
ax = fig.add_subplot(111)
# 添加标题
ax.set_title('散点图')
plt.xlabel('每周玩游戏时间百分比')
plt.ylabel('每周消耗冰激凌的升数')
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# 散点图使用datingDataMat矩阵的第二， 第三列数据,并且添加颜色和大小
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# axis()函数给出了形如[xmin,xmax,ymin,ymax]的列表，指定了坐标轴的范围
ax.axis([-2, 25, -0.2, 2.0])

plt.show()
