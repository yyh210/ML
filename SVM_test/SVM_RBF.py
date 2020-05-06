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


def KernelTrans(X, a, kType):  # 'a' is a row vector,KType[1] = √2σ Gauss kernel
    m, n = shape(X)
    k = mat(zeros((m, 1)))
    if kType[0] == 'liner':
        return X * a.T
    else:
        K = exp(sum((X - tile(a, (m, 1))).A**2, 1) / (-1*kType[1]**2))
        return mat(K).T


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, tolerance, KType):  # classlabels is a row vector
        self.X = dataMatIn
        self.C = C
        self.tol = tolerance
        self.m = shape(dataMatIn)[0]
        self.eCache = mat(zeros((self.m, 2)))
        self.b = 0
        self.alpha = mat(zeros((self.m, 1)))   # column vector
        self.labelMat = mat(classLabels)  # column vector
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            tem = KernelTrans(self.X, self.X[i, :], KType)
            self.K[:, i] = tem

def calEi(oS, i):
    fxi = float(multiply(oS.alpha, oS.labelMat).T * oS.K[:, i]) + oS.b  # calculate f(xi)
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
    eta = oS.K[i, i] + oS.K[j, j] - 2.0 * oS.K[i, j]
    oS.alpha[j] += oS.labelMat[j] * (Ei - Ej) / eta
    oS.alpha[j] = clipAlpha(oS.alpha[j], H, L)
    updateEk(oS, j)
    if abs(oS.alpha[j] - alphaJ_old) < 0.00001:  # important have effect on the b
        return 0
    oS.alpha[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJ_old - oS.alpha[j])
    updateEk(oS, i)
    b1 = oS.b - Ei - oS.labelMat[i] * (oS.alpha[i] - alphaI_old) * \
         oS.K[i, i] - oS.labelMat[j] * (oS.alpha[j] - alphaJ_old) * oS.K[i, j]
    b2 = oS.b - Ej - oS.labelMat[i] * (oS.alpha[i] - alphaI_old) * \
         oS.K[i ,j] - oS.labelMat[j] * (oS.alpha[j] - alphaJ_old) * oS.K[j, j]
    if oS.alpha[i] > 0 and oS.alpha[i] < oS.C:
        oS.b = b1
    elif oS.alpha[j] > 0 and oS.alpha[j] < oS.C:
        oS.b = b2
    else:
        oS.b = (b1 + b2) / 2
    return 1


def svm(dataIn, classLabels, C, tol, iterTimesMax, KType):  # dataIn is list, classLabels is 1-dim list
    dataIn = mat(dataIn)
    labelMat = mat(classLabels).transpose()
    oS = optStruct(dataIn, labelMat, C, tol, KType)
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
    # W = oS.X.T * multiply(oS.alpha, oS.labelMat)  # need to cancel
    return oS.alpha, oS.b


def testRbf(k1=1.3): # fe
    data, label = loadDataSet('testSetRBF2.txt')
    alpha, b = svm(data, label, 0.6, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    svIndex = nonzero(alpha.A)[0]
    sVs = dataMat[svIndex]
    svLabels = labelMat[svIndex]
    errorCnt = 0
    m = dataMat.shape[0]
    for i in range(m):
        kernelValue = KernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelValue.T * multiply(alpha[svIndex], svLabels) + b
        if sign(predict) != labelMat[i]: errorCnt += 1
    print('the training error rate is %f'%(float(errorCnt)/m))
    print(svIndex, b)


testRbf()
