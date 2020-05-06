import numpy as np
import csv
from sklearn.cluster import KMeans
def cal_CS(data, label, k):
    dr = {}
    x_mean = np.zeros((len(label), 2))
    for i in range(k):
        v = np.nonzero(label == i)[0]
        dr[i] = v
    for i in range(data.shape[0]):
        bi = []
        vi = dr[label[i]]
        vi = vi[vi != i]
        dist = np.sqrt(np.sum((data[vi, :] - data[i, :])**2, 1))
        x_mean[i, 0] = np.mean(dist)
        for j in range(k):
            if j == label[i]:
                continue
            bv = dr[j]
            dist_b = np.sqrt(np.sum((data[bv, :] - data[i, :]) ** 2, 1))
            bi.append(np.mean(dist_b))
        x_mean[i, 1] = np.min(bi)
    return np.mean((x_mean[:, 1] - x_mean[:, 0]) / np.max(x_mean, 1))

def loadDataSet(filename):
    csv_reder = csv.reader(open(filename, encoding='utf-8'))
    data = []
    for row in csv_reder:
        data.append(row)
    data = np.array(data).astype(float)
    return data

def norm(d):
    min_v = np.min(d, 0)
    max_v = np.max(d, 0)
    dw = max_v - min_v
    return (d - min_v)/dw

data = loadDataSet('Wine.csv')
data = norm(data)
l = np.arange(5)+2
for i in l:
    yp = KMeans(n_clusters=i, init='k-means++').fit_predict(data)
    m = cal_CS(data,yp, i)
    #m = metrics.calinski_harabaz_score(data, yp)
    print('class:%d, m value :%f\n' %(i, m))
