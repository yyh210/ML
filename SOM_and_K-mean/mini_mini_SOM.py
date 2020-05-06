import numpy as np
import matplotlib.pyplot as plt
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

class SOM:
    def __init__(self,
                 data,
                 N,
                 M,
                 x_dim,
                 sigma,
                 iterTime,
                 alpha,
                 k=None,
                 k_iter=None):
        self.data = np.array(data).astype(float)
        self.grid = np.random.rand(N * M, x_dim)  # at least 5*sqrt(N) neure
        self.sigma = sigma
        self.iterTime = iterTime
        self.M = M
        self.N = N
        self.alpha = alpha  # 学习率
        self.k = k
        self.k_iter = k_iter

    def dataNormalizaton(self): # 非 one-hot 标准化
        max_feature = np.max(self.data, 0)
        min_feature = np.min(self.data, 0)
        self.data = (self.data - min_feature) / (max_feature - min_feature)

    def rbf(self, j_bmu, j):
        dj = self.grid[j_bmu, :] - self.grid[j, :]
        norm_2 = np.sqrt(np.sum(2*dj * dj))
        return np.exp(-norm_2/(self.sigma * self.sigma))


    def chooseBMU(self, xi): #xi is index
        d_M = self.grid - self.data[xi, :]  # 计算输入向量xi和所有网格的距离
        norm2_MM = np.sum(d_M*d_M, 1)   # 所有向量二范数的平方和向量
        sort_index = np.argsort(norm2_MM)
        BMU = sort_index[0]
        return BMU

    def train(self):
        for i in range(self.iterTime):
            if self.sigma > 0.05:  # 更新sigma alpha 并且保证参数有下届
                self.sigma *= 0.9
            if self.alpha > 0.5:
                self.alpha *= 0.9
            for xi_index in range(self.data.shape[0]):
                xi = self.data[xi_index, :]
                bmu = self.chooseBMU(xi_index)
                for grid_id in range(self.M * self.N):  # 更新优胜邻域
                    yk = self.grid[grid_id, :]
                    dw = self.alpha * self.rbf(bmu, grid_id) * (xi - yk)
                    self.grid[grid_id, :] = yk + dw

    def cluster(self):
        grid_label = k_means(self.k, self.grid, self.k_iter)
        x_label = np.zeros(self.data.shape[0])
        for i in range(self.data.shape[0]):
            bmu = self.chooseBMU(i)
            x_label[i] = grid_label[bmu]

        return x_label

    def classify(self):
        good = self.label == 1
        good = np.nonzero(good)[0]
        g = {}
        for i in good:
            bmu = self.chooseBMU(i)
            if bmu not in g.keys():
                g[bmu] = 0
            g[bmu] += 1
        b = {}
        bad = self.label == 0
        bad = np.nonzero(bad)[0]
        for i in bad:
            bmu = self.chooseBMU(i)
            if bmu not in b.keys():
                b[bmu] = 0
            b[bmu] += 1
        return g, b


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


data_in = loadDataSet('Wine.csv')
data_in = norm(data_in)
lab = k_means(3, data_in, 100)
lab = lab.astype(int)
f = open('res', 'w')
np.savetxt(f, lab, delimiter='\n', fmt='%d')

