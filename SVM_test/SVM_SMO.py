from numpy import *


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        dataLine = line.strip().split('\t')
        labelMat.append(float(dataLine[-1]))
        dataMat.append([float(dataLine[0]), float(dataLine[1])])

    return dataMat, labelMat


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, tolerance):  # classlabels is a row vector
        self.X = dataMatIn
        self.C = C
        self.tol = tolerance
        self.m = shape(dataMatIn)[0]
        self.eCache = mat(zeros((self.m, 2)))
        self.b = 0
        self.alpha = mat(zeros((self.m, 1)))   # column vector
        self.labelMat = mat(classLabels)  # column vector


def calEi(oS, i):
    fxi = float(multiply(oS.alpha, oS.labelMat).T * (oS.X * oS.X[i, :].T)) + oS.b  # calculate f(xi)
    Ei = fxi - float(oS.labelMat[i])
    return Ei


def selectJ(oS, i):
    Ei = calEi(oS, i)
    oS.eCache[i] = [1, Ei]
    validError = nonzero(oS.eCache[:, 0].A)[0]
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    if len(validError) > 1:
        for k in validError:
            if k == i:
                continue
            Ek = calEi(oS, k)
            delta = abs(Ek - Ei)
            if maxDeltaE < delta:
                maxDeltaE = delta
                Ej = Ek
                maxK = k
        return maxK, Ej
    else:
        j = i
        while j == i:
            j = int(oS.m*random.rand())
        Ej = calEi(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calEi(oS, k)
    oS.eCache[k] = [1, Ek]
    return

def clipAlpha(aj, high, low):
    if aj > high:
        aj = high
    if aj < low:
        aj = low
    return aj


def innerLoop(oS, i):
    Ei = calEi(oS, i)
    if ( oS.labelMat[i]*Ei< -oS.tol and oS.alpha[i] < oS.C) or \
            ( oS.labelMat[i]*Ei > oS.tol and oS.alpha[i] > 0):
        j, Ej = selectJ(oS, i)
    else: return 0
    alphaJ_old = oS.alpha[j].copy()
    alphaI_old = oS.alpha[i].copy()
    if oS.labelMat[i] != oS.labelMat[j]:
        L = max(0, oS.alpha[j] - oS.alpha[i])
        H = min(oS.C, oS.C + oS.alpha[j] - oS.alpha[i])
    else:
        L = max(0, oS.alpha[j] + oS.alpha[i] - oS.C)
        H = min(oS.C, oS.alpha[j] + oS.alpha[i])
    if L == H:
        return 0
    eta = oS.X[i, :] * oS.X[i, :].T + oS.X[j, :] * oS.X[j, :].T - 2.0 * oS.X[i, :] * oS.X[j, :].T
    oS.alpha[j] += oS.labelMat[j] * (Ei - Ej) / eta
    oS.alpha[j] = clipAlpha(oS.alpha[j], H, L)
    updateEk(oS, j)
    if abs(oS.alpha[j] - alphaJ_old) < 0.00001:  # important have effect on the b
        return 0
    oS.alpha[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJ_old - oS.alpha[j])
    updateEk(oS, i)
    b1 = oS.b - Ei - oS.labelMat[i] * (oS.alpha[i] - alphaI_old) * \
         oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * \
         (oS.alpha[j] - alphaJ_old) * oS.X[i, :] * oS.X[j, :].T
    b2 = oS.b - Ej - oS.labelMat[i] * (oS.alpha[i] - alphaI_old) * \
         oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * \
         (oS.alpha[j] - alphaJ_old) * oS.X[j, :] * oS.X[j, :].T
    if oS.alpha[i] > 0 and oS.alpha[i] < oS.C:
        oS.b = b1
    elif oS.alpha[j] > 0 and oS.alpha[j] < oS.C:
        oS.b = b2
    else:
        oS.b = (b1 + b2) / 2
    return 1


def svm(dataIn, classLabels, C, tol, iterTimesMax):
    dataIn = mat(dataIn)
    labelMat = mat(classLabels).transpose()
    oS = optStruct(dataIn, labelMat, C, tol)
    iterTimes = 0
    alphaPairChanged = 0
    entireSet = True
    while iterTimes < iterTimesMax and ((alphaPairChanged > 0) or entireSet):
        alphaPairChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairChanged += innerLoop(oS, i)
        else:
            non_bound = nonzero((oS.alpha.A > 0) * (oS.alpha.A < oS.C))[0]    # update on the entire set is slow
            for i in non_bound:
                alphaPairChanged += innerLoop(oS, i)
        iterTimes += 1
        if entireSet:
            entireSet = False
        elif alphaPairChanged == 0:
            entireSet = True
    W = oS.X.T * multiply(oS.alpha, oS.labelMat)
    return oS.alpha, oS.b, W


data, label = loadDataSet('test.txt')
Alpha, B, w = svm(data, label, 0.6, 0.001, 200)

print(w, '\n', B)



