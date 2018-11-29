# -*- coding:utf-8 -*-

import codecs

import numpy as np
import win_unicode_console

win_unicode_console.enable()

DATA = open('./eminem1.txt', 'r', encoding='utf-8').read()
T_SIZE = len(DATA)
WORD = list(set(DATA))
V_SIZE = len(WORD)
H_SIZE = 100
Z_SIZE = H_SIZE + V_SIZE
STEP = 25
LEARNING_RATE = 0.1
EPSILON = 1e-8
TRAIN_ITER = 100000
PRINT_ITER = 100
GEN_LEN = 20
SAMPLE_INTER = 300


def num2one_hot(n):
    targets = np.array([n]).reshape(-1)
    d = np.eye(V_SIZE)[targets]
    return d.T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


def dtanh(x):
    return 1 - x ** 2


def softmax(a):
    m = np.max(a)
    s = np.sum(np.exp(a - m))
    return np.exp(a - m) / s


def sample():
    th_prev = H_PREV
    tc_prev = C_PREV
    v = np.random.randint(0, V_SIZE)
    input = num2one_hot(v)
    w = [INT2WORD[v]]
    for i in range(GEN_LEN):
        z = np.row_stack((input, th_prev))
        f = sigmoid(np.dot(W_F, z) + B_F)
        c_bar = np.tanh(np.dot(W_C, z) + B_C)
        i = sigmoid(np.dot(W_I, z) + B_I)
        c = f * tc_prev + i * c_bar
        tc_prev = c
        o = sigmoid(np.dot(W_O, z) + B_O)
        h = np.tanh(c)*o
        th_prev = h
        v = np.dot(W_V, h) + B_V
        y_hat = softmax(v)
        t = np.argmax(y_hat)
        input = num2one_hot(t)
        w.append(INT2WORD[t])
    print(''.join([u for u in w]))


W_F = 0.5*np.random.rand(H_SIZE, Z_SIZE)
B_F = 0.5*np.random.rand(H_SIZE, 1)
W_C = 0.5*np.random.rand(H_SIZE, Z_SIZE)
B_C = 0.5*np.random.rand(H_SIZE, 1)
W_I = 0.5*np.random.rand(H_SIZE, Z_SIZE)
B_I = 0.5*np.random.rand(H_SIZE, 1)
W_O = 0.5*np.random.rand(H_SIZE, Z_SIZE)
B_O = 0.5*np.random.rand(H_SIZE, 1)
W_V = 0.5*np.random.rand(V_SIZE, H_SIZE)
B_V = 0.5*np.random.rand(V_SIZE, 1)

WORD2INT = {v: i for i, v in enumerate(WORD)}
INT2WORD = {i: v for i, v in enumerate(WORD)}


C_PREV = np.zeros((H_SIZE, 1), dtype=np.float)
H_PREV = np.zeros((H_SIZE, 1), dtype=np.float)
D_C_NEXT = np.zeros((H_SIZE, 1), dtype=np.float)
D_H_NEXT = np.zeros((H_SIZE, 1), dtype=np.float)

W_F_G = 0.0
B_F_G = 0.0
W_C_G = 0.0
B_C_G = 0.0
W_I_G = 0.0
B_I_G = 0.0
W_O_G = 0.0
B_O_G = 0.0
W_V_G = 0.0
B_V_G = 0.0


def load():
    global W_F, B_F, W_C, B_C, W_I, B_I, W_O, B_O, W_V, B_V, WORD2INT, INT2WORD
    base = './poem/'
    with codecs.open(base + 'WORD2INT.txt', 'w', 'utf-8') as file:
        while True:
            line = file.readline().strip()
            if not line:
                break
            ds = line.split(':')
            WORD2INT[ds[0]] = int(ds[1])
            INT2WORD[int(ds[1])] = ds[0]
    W_F = np.load(base + 'W_F.npy')
    B_F = np.load(base + 'B_F.npy')
    W_C = np.load(base + 'W_C.npy')
    B_C = np.load(base + 'B_C.npy')
    W_I = np.load(base + 'W_I.npy')
    B_I = np.load(base + 'B_I.npy')
    W_O = np.load(base + 'W_O.npy')
    B_O = np.load(base + 'B_O.npy')
    W_V = np.load(base + 'W_V.npy')
    B_V = np.load(base + 'B_V.npy')


def save():
    global W_F, B_F, W_C, B_C, W_I, B_I, W_O, B_O, W_V, B_V
    base = './poem/'
    s = ''.join([(k + ':' + str(WORD2INT[k]) + '\n') for k in WORD2INT])
    with codecs.open(base + 'WORD2INT.txt', 'w', 'utf-8') as file:
        file.write(s)
    np.save(base + 'W_F', W_F)
    np.save(base + 'B_F', B_F)
    np.save(base + 'W_C', W_C)
    np.save(base + 'B_C', B_C)
    np.save(base + 'W_I', W_I)
    np.save(base + 'B_I', B_I)
    np.save(base + 'W_O', W_O)
    np.save(base + 'B_O', B_O)
    np.save(base + 'W_V', W_V)
    np.save(base + 'B_V', B_V)


P = 0
for iter in range(TRAIN_ITER):
    if iter % SAMPLE_INTER == 0:
        sample()
    if P > T_SIZE:
        P = 0
    expect_end = P + STEP
    end = expect_end
    if expect_end >= len(DATA):
        end = len(DATA) - 1
    x = DATA[P: end]
    y = DATA[P + 1: end + 1]
    f_m, c_bar_m, i_m, c_m, o_m, v_m, y_m, h_m, z_m = {}, {}, {}, {}, {}, {}, {}, {}, {}
    loss = 0.0
    for k, v in enumerate(x):
        input_num = WORD2INT[v]
        target_num = WORD2INT[y[k]]
        input_v = num2one_hot(input_num)
        z = np.row_stack((input_v, H_PREV))
        z_m[k] = z
        f = sigmoid(np.dot(W_F, z) + B_F)
        f_m[k] = f
        c_bar = np.tanh(np.dot(W_C, z) + B_C)
        c_bar_m[k] = c_bar
        i = sigmoid(np.dot(W_I, z) + B_I)
        i_m[k] = i
        c = f * C_PREV + i * c_bar
        c_m[k] = c
        C_PREV = c
        o = sigmoid(np.dot(W_O, z) + B_O)
        o_m[k] = o
        h = np.tanh(c) * o
        h_m[k] = h
        H_PREV = h
        v = np.dot(W_V, h) + B_V
        v_m[k] = v
        y_hat = softmax(v)
        y_m[k] = y_hat
        loss += -np.log(y_hat[target_num] + 0.0001)
    if iter % PRINT_ITER == 0:
        print('%d/%d---loss: %f' % (iter, TRAIN_ITER, loss))
    d_wf = np.zeros_like(W_F)
    d_bf = np.zeros_like(B_F)
    d_wc = np.zeros_like(W_C)
    d_bc = np.zeros_like(B_C)
    d_wi = np.zeros_like(W_I)
    d_bi = np.zeros_like(B_I)
    d_wo = np.zeros_like(W_O)
    d_bo = np.zeros_like(B_O)
    d_wv = np.zeros_like(W_V)
    d_bv = np.zeros_like(B_V)
    for k in range(len(x) - 1, -1, -1):
        target = num2one_hot(WORD2INT[y[k]])
        # e = y_m[k] - target  # V_SIZE x 1
        d_v = y_m[k] - target  # V_SIZE x 1
        d_wv += np.dot(d_v, h_m[k].T)  # V_SIZE x H_SIZE
        d_bv += d_v  # V_SIZE x 1
        d_h = np.dot(W_V.T, d_v) + D_H_NEXT  # H_SIZE x 1
        d_o = d_h * np.tanh(c_m[k])  # H_SIZE x 1
        d_c = d_h * o_m[k] * (1 - np.tanh(c_m[k]) ** 2) + D_C_NEXT  # H_SIZE x 1
        d_wo += np.dot(d_o * o_m[k] * (1 - o_m[k]), z_m[k].T)  # H_SIZE x Z_SIZE
        d_bo += d_o * o_m[k] * (1 - o_m[k])  # H_SIZE x 1
        d_f = d_c * c_bar_m[k]  # H_SIZE x 1
        d_i = d_c * c_bar_m[k]  # H_SIZE x 1
        d_wi += np.dot(d_i * i_m[k]*(1 - i_m[k]), z_m[k].T)
        d_bi += d_i * i_m[k]*(1 - i_m[k])
        d_c_bar = d_c * i_m[k]  # H_SIZE x 1
        d_wc += np.dot(d_c_bar * (1 - c_bar_m[k] ** 2), z_m[k].T)  # H_SIZE x Z_SIZE
        d_bc += d_c_bar * (1 - c_bar_m[k] ** 2)  # H_SIZE x 1
        d_wf += np.dot(d_f * f_m[k] * (1 - f_m[k]), z_m[k].T)  ## H_SIZE x Z_SIZE
        d_bf += d_f * f_m[k]*(1 - f_m[k])  # H_SIZE x 1
        d_z = np.dot(W_F.T, d_f * f_m[k] * (1 - f_m[k])) + np.dot(W_C.T, d_c_bar * (1 - c_bar_m[k] ** 2)) + \
              np.dot(W_I.T, d_i * i_m[k] * (1 - i_m[k])) + np.dot(W_O.T, d_o * o_m[k] * (1 - o_m[k]))
        D_H_NEXT = d_z[V_SIZE:, :]
        D_C_NEXT = d_c * f_m[k]
    wf_update = np.clip(d_wf, -0.01, 0.01)
    W_F_G += wf_update * wf_update
    W_F -= LEARNING_RATE*wf_update/(np.sqrt(W_F_G+EPSILON))
    bf_update = np.clip(d_bf, -1, 1)
    B_F_G += bf_update * bf_update
    B_F -= LEARNING_RATE*bf_update/(np.sqrt(B_F_G+EPSILON))

    wc_update = np.clip(d_wc, -1, 1)
    W_C_G += wc_update * wc_update
    W_C -= LEARNING_RATE * wc_update / (np.sqrt(W_C_G + EPSILON))
    bc_update = np.clip(d_bc, -1, 1)
    B_C_G += bc_update * bc_update
    B_C -= LEARNING_RATE * bc_update / (np.sqrt(B_C_G + EPSILON))

    wi_update = np.clip(d_wi, -1, 1)
    W_I_G += wi_update * wi_update
    W_I -= LEARNING_RATE * wi_update / (np.sqrt(W_I_G + EPSILON))
    bi_update = np.clip(d_bi, -1, 1)
    B_I_G += bi_update * bi_update
    B_I -= LEARNING_RATE * bi_update / (np.sqrt(B_I_G + EPSILON))

    wo_update = np.clip(d_wo, -1, 1)
    W_O_G += wo_update * wo_update
    W_O -= LEARNING_RATE * wo_update / (np.sqrt(W_O_G + EPSILON))
    bo_update = np.clip(d_bo, -1, 1)
    B_O_G += bo_update * bo_update
    B_O -= LEARNING_RATE * bo_update / (np.sqrt(B_O_G + EPSILON))

    wv_update = np.clip(d_wv, -1, 1)
    W_V_G += wv_update * wv_update
    W_V -= LEARNING_RATE * wv_update / (np.sqrt(W_V_G + EPSILON))
    bv_update = np.clip(d_bv, -1, 1)
    B_V_G += bv_update * bv_update
    B_V -= LEARNING_RATE * bv_update / (np.sqrt(B_V_G + EPSILON))

    P += STEP
    save()

