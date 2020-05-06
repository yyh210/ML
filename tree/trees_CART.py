from math import log
import operator


def splitDataSet(dataSet, axis, value):
    subData = []
    for featVec in dataSet:
        if featVec[axis] == value:
            remVec = featVec[:axis]
            remVec.extend(featVec[axis+1:])
            subData.append(remVec)
    return subData


def chooseBestFeature(dataSet):
    numFeature = len(dataSet[0]) - 1
    minGiniIndex = 5.0
    bestFeature = -1
    for i in range(numFeature):
        featureValue = [example[i] for example in dataSet]  # get all the feature-values on dimension i.
        uniqueValues = set(featureValue)
        newGiniIndex = 0.0
        for value in uniqueValues:
            subDataSet = splitDataSet(dataSet, i, value)
            frequency = len(subDataSet)/float(len(dataSet))
            subGiniIndex = calcGiniIndex(subDataSet)
            newGiniIndex += frequency * subGiniIndex
        if newGiniIndex < minGiniIndex:
            minGiniIndex = newGiniIndex
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


# compute Gini index
def calcGiniIndex(dataSet):
    labelsCount = {}
    labelsList = [example[-1] for example in dataSet]
    numElements = len(labelsList)
    giniIndex = 1.0
    for value in labelsList:
        if value not in labelsCount.keys():
            labelsCount[value] = 0
        labelsCount[value] += 1
    for key in labelsCount:
        prob = float(labelsCount[key])/numElements
        giniIndex -= prob**2
    return giniIndex


