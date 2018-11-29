"""
RNNs
"""
# -*- coding:utf-8 -*-

import numpy as np
import copy
import math
import codecs

# np.random.seed(0)

C = 10000
MAX_NUM = pow(2, 8)
BINARY = np.unpackbits(np.array([range(MAX_NUM)], dtype=np.uint8).T, axis=1)
INT2BINARY= {}

RATE = 0.1
W_XH = 2*np.random.random([2, 16])-1    #  [-1, 1)
W_HH = 2*np.random.random([16, 16])-1
W_HY = 2*np.random.random([16, 1])-1
L0 = np.zeros(16)

for i in range(MAX_NUM):
    INT2BINARY[i] = BINARY[i]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_drv(x):
    return x*(1-x)


def cos(a, b):
    p = np.dot(a, b.T)
    sum = 0.0
    for i in a:
        sum += i**2
    sqrt_a = math.sqrt(sum)
    sum = 0.0
    for i in b:
        sum += i**2
    sqrt_b = math.sqrt(sum)
    return p / (sqrt_a*sqrt_b)
        


w_hh_update = np.zeros_like(W_HH)
w_xh_update = np.zeros_like(W_XH)
w_hy_update = np.zeros_like(W_HY)
def train():
    global W_HH, W_XH, W_HY, CURRENT_Y, CURRENT_X, X, LABELS, RATE, w_hh_update, w_hy_update, w_xh_update
    s = ''
    for i in range(C):
        m = np.random.randint(MAX_NUM/2)
        n = np.random.randint(MAX_NUM/2)
        current_x1 = INT2BINARY[m]
        current_x2 = INT2BINARY[n]
        c = m + n
        current_y = INT2BINARY[c]
        r = np.zeros_like(current_y)

        e_drv_layer2_in_list = list()
        layer_1_list = list()
        layer_1_list.append(L0)
        
        for j in range(8):
            x = np.array([[current_x1[8-j-1], current_x2[8-j-1]]])  # 1*2
            y = np.array([[current_y[8-j-1]]]).T  # 1x1
            layer_1 = sigmoid(np.dot(x, W_XH) + np.dot(layer_1_list[-1], W_HH))  # 1x16
            layer_1_list.append(copy.deepcopy(layer_1))
            layer_2 = sigmoid(np.dot(layer_1, W_HY))
            r[8-j-1] = np.round(layer_2[0][0])
            e_drv_layer2_out = layer_2 - y
            e_drv_layer2_in = e_drv_layer2_out*sigmoid_drv(layer_2)  # 1x1
            e_drv_layer2_in_list.append(e_drv_layer2_in)
        next_err_drv_layer1_in_list = list()
        next_err_drv_layer1_in_list.append(np.zeros((1, 16)))
        # next_err_drv_layer1_in = np.zeros((1, 16)
        for j in range(8):
            _x = np.array([[current_x1[j], current_x2[j]]])  # 1x2
            layer_1 = layer_1_list[-j-1]
            prev_layer_1 = layer_1_list[-j-2]
            e_drv_layer2_in = e_drv_layer2_in_list[-j-1]  # 1x1

            next_err_drv_sum = np.zeros((1, 16))
            next_err_drv_list = np.zeros(16)
            if j == 0:
                next_err_drv_sum = next_err_drv_sum.dot(W_HH.T)
            else:
                for k, v in enumerate(next_err_drv_layer1_in_list):
                    m = k
                    if m < j:
                        base = next_err_drv_layer1_in_list[m].dot(W_HH.T)
                        next_rs = base
                        while m < j - 1:
                           next_rs = (next_rs*sigmoid_drv(layer_1_list[-m-2])).dot(W_HH.T)
                           m += 1
                        next_err_drv_sum += next_rs

            e_drv_layer1_in = (next_err_drv_sum + e_drv_layer2_in.dot(W_HY.T))*sigmoid_drv(layer_1)
            next_err_drv_layer1_in_list.append(e_drv_layer1_in)
            w_hy_update += np.atleast_2d(layer_1).T.dot(e_drv_layer2_in)
            w_hh_update += np.atleast_2d(prev_layer_1).T.dot(e_drv_layer1_in)
            w_xh_update += _x.T.dot(e_drv_layer1_in)
            
        W_XH -= w_xh_update*RATE
        W_HH -= w_hh_update*RATE
        W_HY -= w_hy_update*RATE
        # print(W_XH, W_HH, W_HY)

        w_xh_update *= 0.0
        w_hh_update *= 0.0
        w_hy_update *= 0.0

        sm = cos(r, current_y)
        s += 'pred:' + str(r) + "\treal:" + str(current_y) + "\tsimilarity:" + str(sm) + '\n'
        if i % 100 == 0:
            print('pred:', r)
            print('real:', current_y)
            print('similarity:', sm)
            print('=====')
    with codecs.open('rnn_rs2.txt', mode='w', encoding='utf-8') as file:
        file.write(s)


if __name__ == '__main__':
    train()
