from numpy import *
import operator


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqdistances = sqDiffMat.sum(1)
    distances = sqdistances**0.5
    sortedDistancesIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistancesIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), \
                              reverse=True)

    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numOfArray = len(arrayLines)
    returnMat = zeros((numOfArray, 2))
    classLabelVector = []
    index = 0
    for element in arrayLines:
        element = element.strip().split(' ')
        classLabelVector.append(int(element[-1]))
        returnMat[index, :] = element[0:-1]
        index += 1
    return returnMat, array(classLabelVector)


def autoNorm(dataSet):
    minvalue = dataSet.min(0)
    maxvalue = dataSet.max(0)
    ranges = maxvalue - minvalue
    normDataSet = zeros(dataSet.shape)  # initialize
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minvalue, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minvalue


def dateClassifyTest():
    testRatio = 0.1
    dataMat, dataLabels = file2matrix('watermelon.txt')
    normMat, b, c = autoNorm(dataMat)
    totalNum = dataMat.shape[0]
    testNum = int(testRatio * totalNum)
    errorNum = 0.0
    for k in arange(3)+1:
        cnt = 0
        for i in arange(dataMat.shape[0]):
            t_label = dataLabels[i]
            p_label = classify0(dataMat[i, :], dataMat, dataLabels, k)
            if t_label == p_label:
                cnt += 1
        print('this k is %d and %d inputs are classified' % (k, cnt))



print('loaded')
dateClassifyTest()
