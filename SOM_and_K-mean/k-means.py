import numpy as np
import csv

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

def k_means(k, dataIn, iter_time):
    # 基于欧拉距离
    # 随机选择k个centroid构成质心集合MU
    # 遍历数据集，对每一个输入xi进行分类
    # 计算每个簇的质心，如果新的质心 != 旧质心 更新质心
    # 回到line 18 || 达到停机条件 退出
    # 返回聚类标签向量
    dataIn = np.array(dataIn)
    dataIn = dataIn.astype(float)
    m, n = np.shape(dataIn)
    perm = np.arange(m)
    np.random.shuffle(perm)
    mu = dataIn[perm[:k], :]
    label = np.zeros(m)
    for j in range(iter_time):
        for i in range(m):
            xi = dataIn[i, :]
            # 挑选最小的 距离进行聚类
            min_mu = 0
            min_dist = np.sqrt(np.sum((xi - mu[0, :])**2))
            for mu_i in range(k):
                temp = np.sqrt(np.sum(np.power((xi - mu[mu_i, :]), 2)))
                if temp < min_dist:
                    min_mu = mu_i
                    min_dist = temp
            label[i] = min_mu
        # 更新质心
        for mu_i in range(k):
            ci_index = (label == mu_i)
            ci = dataIn[ci_index, :]
            if len(ci) != 0:
                ci_mean = np.mean(ci, 0)
                mu[mu_i, :] = ci_mean
    return label


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

    return (x_mean[:, 1] - x_mean[:, 0]) / np.max(x_mean, 1)

def show(data):
    l = np.arange(5)+2
    for i in l:
        y = k_means(i, data, 40)
        m = np.mean(cal_CS(data, y, i))
        print('k :%d  m value:%f'%(i, m))


data_in = loadDataSet('Wine.csv')
data_in = norm(data_in)
show(data_in)
#最优K值为3，以下为保存分类结果。
#lab = k_means(3, data_in, 100)
#lab = lab.astype(int)
#f = open('res', 'w')
#np.savetxt(f, lab, delimiter='\n', fmt='%d')
