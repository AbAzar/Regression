import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split
ailerons = pd.read_csv('ailerons.dat', header=-1)
diabetes = pd.read_csv('diabetes.dat', header=-1)
house = pd.read_csv('house.dat', header=-1)
# pumadyn_32fm = pd.read_csv('pumadyn-32fm.data', delim_whitespace=True, header=-1)
# pumadyn_32nh = pd.read_csv('pumadyn-32nh.data', delim_whitespace=True, header=-1)


def lin_reg(x, y):

    tmp = np.linalg.pinv(np.matmul(x.transpose(), x))
    w = np.matmul(np.matmul(tmp, x.transpose()), y)
    return w


def predict(x, w):
    W = np.matmul(x, w)
    return W


def ri_reg(lnd, x, y):

    tmp = np.linalg.pinv(np.matmul(x.transpose(), x) + np.multiply(lnd, np.identity(num_col-1)))
    w = np.matmul(np.matmul(tmp, x.transpose()), y)
    return w


def soft(a, dlt):
    sign = 1
    if a < 0:
        sign = -1
    return max(abs(a)-dlt, 0)*sign


def lasso_reg(lnd, x, y):
    tmp = np.linalg.pinv(np.matmul(x.transpose(), x) + np.multiply(lnd, np.identity(num_col-1)))
    ini_w = np.matmul(np.matmul(tmp, x.transpose()), y)
    i = 0
    w = ini_w.copy()
    while i < 10:
        for j in range(num_col-1):
            a = np.power(x[:, j], 2).sum() * 2
            if a == 0:
                return False
            c1 = np.matmul(x, ini_w).ravel()
            c2 = np.multiply(ini_w[j], x[:, j]).ravel()
            c = np.dot(x[:, j], y.ravel() - c1 + c2).sum() * 2
            w[j] = soft(c/a, lnd/a)
        ini_w = w.copy()
        i += 1
    return w


def sqt_err(ype, yte):
    return np.power(ype.ravel()-yte.ravel(), 2).sum() / len(yte)


def kernel_re_reg(x, y, lnd, metric='rbf', params={'gamma': 0.5}):
    k = pairwise_kernels(x, None, metric=metric, filter_params=True, **params)
    alpha = np.matmul(np.linalg.pinv(k + np.multiply(lnd, np.identity(len(y)))), y)
    return alpha


def predict_k(x_test, x_train, alpha, metric='rbf', params={"gamma": 0.5}):
    k = pairwise_kernels(x_test, x_train, metric=metric, filter_params=True, **params)
    return np.dot(k, alpha)


data = ailerons
num_col = data.shape[1]
num_rw = data.shape[0]
x = data.iloc[:, 0:num_col - 1].as_matrix()
y = data.iloc[:, num_col - 1:num_col].as_matrix()


s = 0
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    w = lin_reg(x_train, y_train)
    ype = predict(x_test, w)
    s += sqt_err(ype, y_test)
print(s / 10)

s = [0, 0, 0, 0, 0]
lnds = [0.5, 1, 10, 100, 1000]
for l in range(len(lnds)):
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
        w = ri_reg(lnds[l], x_train, y_train)
        ype = predict(x_test, w)
        s[l] += sqt_err(ype, y_test)
    s[l] = s[l] / 10
print(s)

s = [0, 0, 0, 0, 0]
lnds = [0.5, 1, 10, 100, 1000]
cnt = 10
for l in range(len(lnds)):
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
        w = lasso_reg(lnds[l], x_train, y_train)
        if w is False:
            cnt -= 1
            continue
        ype = predict(x_test, w)
        s[l] += sqt_err(ype, y_test)
    s[l] = s[l] / cnt
print(s)

lnds = [0.5, 1, 10, 100, 1000]
lnds_d = [(x, y) for x in lnds for y in [2, 5, 10]]
s = [0 for i in range(len(lnds_d))]
for l in range(len(lnds_d)):
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
        alpha = kernel_re_reg(x_train, y_train, lnds_d[l][0], 'polynomial', {'gamma':1, 'coef0':1, 'degree': lnds_d[l][1]})
        ype = predict_k(x_test, x_train, alpha, 'polynomial', {'gamma':1, 'coef0':1, 'degree': lnds_d[l][1]})
        s[l] += sqt_err(ype, y_test)
        print(lnds_d[l], s[l])
    s[l] = s[l] / 10
print(s)

s = [0, 0, 0, 0, 0]
lnds = [0.5, 1, 10, 100, 1000]
for l in range(len(lnds)):
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
        alpha = kernel_re_reg(x_train, y_train, lnds_d[l][0], 'rbf', {'gamma':0.5})
        ype = predict_k(x_test, x_train, alpha, 'rbf', {'gamma':0.5})
        s[l] += sqt_err(ype, y_test)
        print(lnds_d[l], s[l])
    s[l] = s[l] / 10
print(s)


