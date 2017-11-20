# -*- coding: UTF-8 -*-
'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表滥用的， 0代表没有
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


# 创建所有词的set集合
def createVocabList(dataSet):
    # 初始化空的vocabSet
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 求两个集合的并集
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


# vocabList为输入参数，需要判定的数组
# inputSet为所有词的集合
# 如果在 vocabList 中的词在 inputSet 中出现，就在 vocabList 的相应位置置位（置1）
def setOfWords2Vec(vocabList, inputSet):
    # 初始化returnVec，初始值都为0，长度是vocabList的长度
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # 如果 word 出现了，相应的位置置1
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec


# 获取数据集
# listOPosts, listClasses = loadDataSet()
# print listOPosts
# 获取数据set
# myVocabList = createVocabList(listOPosts)
# print myVocabList
# print setOfWords2Vec(myVocabList, listOPosts[0])
# print setOfWords2Vec(myVocabList, listOPosts[1])

# trainCategory: 每篇文档类别标签所构成的 向量trainCategory。 eg: [0, 0, 1],三篇文章，在第三篇文章中出现了
# trainMatrix为词向量，eg[[0, 1, 1, 0, 0, 0, 1, 0, 0, 0],[0, 1, 1, 0, 0, 0, 0, 0, 0, 1],[0, 1, 1, 0, 0, 1, 0, 0, 0, 0]]
# 其中的每个向量表示所有文章的词集合set，如果等于1表示这个词在这个文章出现过，如果等于0表示这个词在这个文章没有出现过
def trainNB0(trainMatrix, trainCategory):
    # 获得文档总数
    numTrainDocs = len(trainMatrix)
    # 获取词集合中词的数量
    numWords = len(trainMatrix[0])
    # 计算概率p(ci)，通过类别中（有敏感词或者没有敏感词）文档数除以总的文档数来计算
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 生成单位矩阵,如果其中一个概率值为0，那么最后的乘积也为0，为了降低这种影响，将所有词的出现数初始化为1，并将分母初始化
    # 为2
    p0Num = ones(numWords)
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0
    p1Denom = 2.0  # change to 2.0
    # 遍历 trainMatrix中 所有文档
    for i in range(numTrainDocs):
        # 第i篇文章被定义为出现了敏感词
        if trainCategory[i] == 1:
            # 一旦词在某一文档中出现，该词对应的个数加1，向量相加
            p1Num += trainMatrix[i]
            # 计算出现此类别的文档的总词数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素做除法，向量除标量
    # 对每个元素做除法，p1Vect向量代表了标签1的所有文章词向量中每个词的概率，p0Vect亦然，
    # pAbusive为标签类别1发生的概率
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


listOPosts, listClasses = loadDataSet()
# 词集合
myVocabList = createVocabList(listOPosts)
# 初始化词集合
trainMat = []
# 把所有文章的词置入到集合trainMat中
for postInDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postInDoc))
# print trainMat
p0v, p1v, pAb = trainNB0(trainMat, listClasses)
print p0v
print p1v
print pAb


# vec2Classify 为词向量化的文章，eg: ['love', 'my', 'dalmation', 'hi', 'nice']为词集合
# 文章为 ['love', 'my', 'dalmation']，则文章词向量化后为 [1, 1, 1, 0, 0]
# p0Vec为标签为0的 词向量概率，eg [0.3, 0.1, 0.1, 0.1, 0.4] 代表了所有标签为0的文章对应的词集合中的概率

# 给定一个词向量化好的文章，断定属于哪个分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 词集合中每个词只能出现一次，有的词在文档中出现多次，所以出现如下的文档词袋模式
# vocabList为词向量，inputSet为需要分析的文档
def bagOfWords2VecMN(vocabList, inputSet):
    # 初始化零向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # 将returnVec对应的词的数量+1，而不是简单的记0 1
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testingNB():
    # 初始化数据集，listOPosts为文章数组， listClasses为标签分类数组，是否包含目标词汇
    listOPosts, listClasses = loadDataSet()
    # 抽取所有文章的词组成集合
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        # 对于每一篇文章，构建对应的词向量，如果 postinDoc的词 在 myVocabList中出现，则将相对应的位置置位
        # 之后追加到trainMat矩阵中
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 训练数据，trainMat 为每篇文章的词向量组成的集合，listClasses 为每篇文章的对应的标签分类数组
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # 测试实体
    testEntry = ['love', 'my', 'dalmation']
    # 构建测试实体的对应的词向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


# testingNB()

# 根据正则构建词向量，分隔符是除单词数字外的任意字符串
def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    # 转换为小写，词袋模型
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    # 邮件文档数组
    docList = []
    # 标签数组
    classList = []
    # 全文数组
    fullText = []
    for i in range(1, 26):
        # 添加垃圾邮件数据训练集
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 添加普通邮件
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 生成词向量
    vocabList = createVocabList(docList)  # create vocabulary
    # 产生1-50数组
    trainingSet = range(50)
    testSet = []  # create test set
    # 随机构建测试数据集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        # 删除抽取出来的训练集的索引
        del (trainingSet[randIndex])
    # 重新构建训练数据集
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        # 构建所有文章的词向量数组
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        # 构建所有文章的标签数组
        trainClasses.append(classList[docIndex])
    # 获取每个标签下的 词向量概率
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 用测试数据集验证准确率,循环遍历测试数据集的索引,之后去最开始初始化的数组寻找具体数据
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ', float(errorCount) / len(testSet)
    # return vocabList,fullText


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = [];
    classList = [];
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen);
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = [];
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
