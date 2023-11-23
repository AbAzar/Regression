import random
import numpy as np
import math
import matplotlib.pyplot as plt


def predict(x, w, b):
    y = w * x + b
    return y


def myplt(x, y, y_noise, y_predict, color):
    plt.scatter(x, y, s=15, c='k')
    plt.plot(x, y, 'm')
    plt.scatter(x, y_noise, s=15)
    plt.plot(x, y_predict, color)
    return plt.show()


def norm2(x, y_p):
    x = np.column_stack((x, np.zeros(np.shape(x)[0]) + 1))
    tmp = np.linalg.pinv(np.matmul(x.transpose(), x))
    w = np.matmul(np.matmul(tmp, x.transpose()), y_p)
    return w[0], w[1]


def norm0(x, y_p):
    m = []
    intercept = []
    for i in range(len(x)):
        for j in range(i + 1, len(y_p)):
            m.append((y_p[j] - y_p[i]) / (x[j] - x[i]))
            intercept.append(-m[-1] * x[i] + y_p[i])

    k = [0 for i in range(len(m))]
    for tp in range(len(m)):
        for point in range(len(y_p)):
            if math.isclose(y_p[point], m[tp] * x[point] + intercept[tp], abs_tol=10 ** -4):
                k[tp] += 1
    maxk = np.max(k)
    ind = np.argmax(k)
    return m[ind], intercept[ind], maxk


def norm1(x, y_noise):
    w = 2
    w_new = w
    gamma = .00001
    error = np.inf
    while True:
        b_new_vector = y_noise - x * w_new
        b_ies = np.median(b_new_vector)
        y_ies = w_new * x + b_ies  # y in each step
        err = y_ies - y_noise
        norm1_err = np.sum(np.abs(err))

        if norm1_err > error:
            break

        derivation = np.sum(x * np.sign(err))
        w = w_new
        b = b_ies
        error = norm1_err
        if derivation != 0:
            old_derivation = derivation
            w_new = w_new - gamma * derivation
        else:
            w_new = w_new - gamma * old_derivation
    return w, b


def norm_inf(x, y_noise):
    w = 2
    w_new = w
    gamma = .00001
    error = np.inf
    while True:
        b_new_vector = y_noise - x * w_new
        b_ies = (np.min(b_new_vector) + np.max(b_new_vector)) / 2
        y_ies = w_new * x + b_ies  # y in each step
        err = y_ies - y_noise
        norm_err = np.max(np.abs(err))

        if norm_err > error:
            break
        ind_max = np.argmax(err)
        ind_min = np.argmin(err)
        derivation = (x[ind_max] * np.sign(err[ind_max]) + x[ind_min] * np.sign(err[ind_min])) / 2
        w = w_new
        b = b_ies
        error = norm_err
        if derivation != 0:
            old_derivation = derivation
            w_new = w_new - gamma * derivation
        else:
            w_new = w_new - gamma * old_derivation
    return w, b


def MSE(y, y_norm):
    return np.power(y.ravel() - y_norm.ravel(), 2).sum() / len(y)


w = 4
b = 4
# n = [2, 20, 50, 100]
n = 2
noise = np.random.normal(0, 10, n)
x = np.array(random.sample(range(1, 200), n))
y = x * w + b
y_p = y + noise
x = np.array(x, np.float64)
y_p = np.array(y_p, np.float64)

w0, b0, l = norm0(x, y_p)
w1, b1 = norm1(x, y_p)
w2, b2 = norm2(x, y_p)
w_inf, b_inf = norm_inf(x, y_p)
print(w0, b0)
print(w1, b1)
print(w2, b2)
print(w_inf, b_inf)

myplt(x, y, y_p, predict(x, w0, b0), 'g')
myplt(x, y, y_p, predict(x, w1, b1), 'g')
myplt(x, y, y_p, predict(x, w2, b2), 'g')
myplt(x, y, y_p, predict(x, w_inf, b_inf), 'g')

y_norm0 = predict(x, w0, b0)
y_norm1 = predict(x, w1, b1)
y_norm2 = predict(x, w2, b2)
y_norm_inf = predict(x, w_inf, b_inf)

print(MSE(y, y_norm0))
print(MSE(y, y_norm1))
print(MSE(y, y_norm2))
print(MSE(y, y_norm_inf))
