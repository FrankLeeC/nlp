# -*- coding: utf-8 -*-

import numpy as np

def num2one_hot(n, size):
    targets = np.array([n]).reshape(-1)
    d = np.eye(size)[targets]
    return d.T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1 - x ** 2


def softmax(a):
    m = np.max(a)
    s = np.sum(np.exp(a - m))
    return np.exp(a - m) / s


def cross_entropy(y, y_hat):
    return -y*np.log(y_hat)


def d_cross_ectropy_softmax(y_hat):
    return y_hat - 1
