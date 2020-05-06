from math import log
import operator


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    subData=[]
    for featVec in dataSet:
        if featVec[axis] == value:
            remVec = featVec[:axis]
            remVec.extend(featVec[axis+1:])
            subData.append(remVec)
    return subData


def chooseBestFeature(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestFeature = -1
    bestInfoGain = 0.0
    for i in range(numFeature):
        featureValue = [example[i] for example in dataSet]  # get all the feature-values on dimension i.
        uniqueValues = set(featureValue)
        newEntropy = 0.0
        for value in uniqueValues:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            subEntropy = calcShannonEnt(subDataSet)
            newEntropy += prob * subEntropy
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeature(dataSet)
    bestLabel = labels[bestFeature]
    featureValueList = [example[bestFeature] for example in dataSet]
    uniqueValueList = set(featureValueList)
    myTree = {bestLabel: {}}
    del(labels[bestFeature])
    for value in uniqueValueList:
        subLabels = labels.copy()
        myTree[bestLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


