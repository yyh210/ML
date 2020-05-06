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


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(m*random.rand())
    return j


def clipAlpha(aj, high, low):
    if aj > high:
        aj = high
    if aj < low:
        aj = low
    return aj

# 简化SMO


def simpleSMO(dataMatIn, classLabels, C, tolerance, maxIter):
    dataMat = mat(dataMatIn)  # [x1',x2',....xn']'
    labelMat = mat(classLabels).transpose()  # column vector
    m = dataMat.shape[0]
    alpha = mat(zeros((m, 1)))  # column vector
    b = 0
    iterationTimes = 0
    while iterationTimes < maxIter:
        alphaChangedTimes = 0
        for i in range(m):
            fxi = float(multiply(alpha, labelMat).T * dataMat * dataMat[i, :].T) + b  # calculate f(xi)
            Ei = fxi - labelMat[i]
            if (Ei*labelMat[i] < -tolerance and alpha[i] < C) or\
                    (Ei*labelMat[i] > tolerance and alpha[i] > 0):   # yi*(f(x) - yi) = yi*f(x) - 1
                j = selectJrand(i, m)
                if labelMat[i] != labelMat[j]:
                    L = max(0, alpha[j]-alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
            else:
                continue
            fxj =float(multiply(alpha, labelMat).T * dataMat * dataMat[j, :].T) + b   # calculate f(xj)
            Ej = fxj - labelMat[j]
            eta = dataMat[i, :]*dataMat[i, :].T + dataMat[j, :]*dataMat[j, :].T - 2.0*dataMat[i, :]*dataMat[j, :].T
            alpha_Jnew = alpha[j] + labelMat[j]*(fxi - labelMat[i] - fxj + labelMat[j]) / eta
            alpha_Jnew = clipAlpha(alpha_Jnew, H, L)
            alpha_Inew = alpha[i] + labelMat[i]*labelMat[j]*(alpha[j] - alpha_Jnew)
            b1_new = -Ei - labelMat[i]*dataMat[i, :]*dataMat[i, :].T * (alpha_Inew - alpha[i]) - \
                labelMat[j]*dataMat[i, :]*dataMat[j, :].T * (alpha_Jnew - alpha[j]) + b
            b2_new = -Ej - labelMat[i]*dataMat[i, :]*dataMat[j, :].T * (alpha_Inew - alpha[i]) - \
                labelMat[j]*dataMat[j, :]*dataMat[j, :].T * (alpha_Jnew - alpha[j]) + b
            # b = (b1_new + b2_new)/2   there art two methods to iterate b

            if alpha_Inew>0 and alpha_Inew<C:    # to identify support vector
                b = b1_new
            elif alpha_Jnew>0 and alpha_Jnew<C:
                b = b2_new
            else:
                b = (b1_new + b2_new)/2.0
            alpha[i] = alpha_Inew
            alpha[j] = alpha_Jnew
            # print('changed')
            alphaChangedTimes += 1

        if alphaChangedTimes == 0:
            iterationTimes += 1
        else:
            iterationTimes = 0
        # print('iteration number is %d', iterationTimes)
    w = dataMat.T * multiply(alpha, labelMat)
    return alpha, b, w


data, label = loadDataSet('test.txt')
Alpha, B, w = simpleSMO(data, label, 0.6, 0.001, 40)

print(w, B)
