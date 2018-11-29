# -*- coding:utf-8 -*-
# import minpy.numpy as np
# import minpy.numpy.random as random
# from minpy.context import cpu, gpu, set_context
import numpy as np
import codecs
import collections
# import math
# import time
import win_unicode_console
import time
win_unicode_console.enable()
# set_context(gpu(0))
# print(gpu(0).device_id, gpu(1).device_id)

PATH = './eminem.txt'
DATASET = []
lines = 0
with codecs.open(PATH, 'r', encoding='utf-8') as file:
    line = ''
    while True:
        line = file.readline().strip()
        if not line:
            break
        else:
            lines += 1
            for i in line:
                DATASET.append(i)
H_SIZE = 130
V_SIZE = set(DATASET).__len__()
Z_SIZE = H_SIZE + V_SIZE
print(V_SIZE)
WORD2INT = {}
INT2WORD = {}
RATE = 0.01
TRAIN_ITER = 5000
PRINT_ITER = 50
SAMPLE_ITER = 100
GEN_LENGTH = 9
GEN_RANGE = 1
min_loss = 1
CLIP_GRAD = True
clip_value = 0.03

count = []
count.extend(collections.Counter(DATASET).most_common(V_SIZE))
for i, v in enumerate(count):
    WORD2INT[v[0]] = i
INT2WORD = dict(zip(WORD2INT.values(), WORD2INT.keys()))

W_F = 2*np.random.rand(H_SIZE, Z_SIZE, )-1
B_F = 2*np.random.rand(H_SIZE, 1)-1
W_C = 2*np.random.rand(H_SIZE, Z_SIZE)-1
B_C = 2*np.random.rand(H_SIZE, 1)-1
W_I = 2*np.random.rand(H_SIZE, Z_SIZE)-1
B_I = 2*np.random.rand(H_SIZE, 1)-1
W_O = 2*np.random.rand(H_SIZE, Z_SIZE)-1
B_O = 2*np.random.rand(H_SIZE, 1)-1
W_V = 2*np.random.rand(V_SIZE, H_SIZE)-1
B_V = 2*np.random.rand(V_SIZE, 1)-1
H_INIT = np.zeros((H_SIZE, 1), dtype=np.float)
C_INIT = np.zeros((H_SIZE, 1), dtype=np.float)
# print(C_INIT)


def word2one_hot(x):
    return num2one_hot(WORD2INT[x])


def num2one_hot(n):
    targets = np.array([n]).reshape(-1)
    d = np.eye(V_SIZE)[targets]
    # ttt = np.sum(d)
    # print(ttt)
    # if ttt[0] == 0:
    #     print('fuck---', ttt, n , INT2WORD[n])

    return d.T


def parse_one_hot(a):
    j = 0
    for i, v in enumerate(a):
        if v[0] == 1:
            j = i
            break
    return INT2WORD[j]


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


def sample(length):
    # np.random.seed(int(time.time()))
    for i in range(GEN_RANGE):
        word = np.random.randint(0, V_SIZE)
        input_one_hot = num2one_hot(word)
        rnn = RNN('', 0)
        rnn.sample(input_one_hot, length)
    

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


def get_grad(m):
    s = np.sum(m)
    return s / (m.shape[0] + m.shape[1])
    # return str(np.max(m)) + '--' + str(np.min(m))


def train():
    n = 0
    c = 0
    # b = False
    count = 0
    for m in range(TRAIN_ITER):
        with codecs.open(PATH, 'r', 'utf-8') as file:
            line = ''
            while True:
                line = file.readline().strip()
                if not line:
                    break
                n += 1
                c += 1
                raw_sentence = []
                for i in line:
                    raw_sentence.append(i)
                sentence = []
                for j, v in enumerate(raw_sentence):
                    sentence.append(word2one_hot(v))
                rnn = RNN(sentence, n)
                rnn.train()
                loss = rnn.get_loss()
                if c % PRINT_ITER == 0:
                    print('%d---%d/%d loss: %f' % (m, c, TRAIN_ITER*lines, float(loss[0][0])))
                    # sample(GEN_LENGTH)
                    g1, g2, g3, g4, g5, g6, g7, g8, g9, g10 = rnn.get_gradient()
                    # print('grad: ', get_grad(g1), get_grad(g2), get_grad(g3), get_grad(g4), get_grad(g5), get_grad(g6), get_grad(g7), get_grad(g8), get_grad(g9), get_grad(g10))
                    print('-------------')
                if c % SAMPLE_ITER == 0:
                    sample(GEN_LENGTH)
                    print('----------')
                if loss < min_loss:
                    print(loss, ' reach min loss: ', min_loss)
                    # b = True
                    sample(GEN_LENGTH)
                #     break
        # if b:
        #     break
    save()
    print(count, '/', TRAIN_ITER*lines)
    sample(GEN_LENGTH)
    print('save over!')


class AdamOptimizer():
    
    def __init__(self, b1, b2, rate, epsilon):
        self.v1 = 0
        self.c1 = 0
        self.b1 = b1
        self.b2 = b2
        self.rate = rate
        self.epsilon = epsilon

    def optimize(self, t, g, x):
        self.v1 = self.b1*self.v1+(1-self.b1)*g
        self.c1 = self.b2*self.c1+(1-self.b2)*(g**2)
        v = self.v1/(1-self.b1**t)
        c = self.c1/(1-self.b2**t)
        x -= self.rate*v/(np.sqrt(c)+self.epsilon)
        return x


optimizer1 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)
optimizer2 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)
optimizer3 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)
optimizer4 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)
optimizer5 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)
optimizer6 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)
optimizer7 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)
optimizer8 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)
optimizer9 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)
optimizer10 = AdamOptimizer(0.9, 0.999, RATE, 1e-8)       


class LSTM():
    
    def __init__(self, x, y, h_prev, c_prev):
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        self.y = y
        # p = 0
        # for i, v in enumerate(self.x):
        #     if v == 1:
        #         p = i
        # ttt = np.sum(self.y)
        # print(ttt[0], '@@@@@@@@@@@@')
        # if (ttt[0] == 0):
        #     print('---------------', p, INT2WORD[p])

    def sample(self):
        global W_F, B_F, W_C, B_C, W_I, B_I, W_O, B_O, W_V, B_V
        z = np.row_stack((self.x, self.h_prev))
        f = sigmoid(np.dot(W_F, z) + B_F)
        c_bar = np.tanh(np.dot(W_C, z) + B_C)
        o = sigmoid(np.dot(W_O, z) + B_O)
        i = sigmoid(np.dot(W_I, z) + B_I)
        c = f * self.c_prev + i * c_bar
        h = np.tanh(c) * o
        v = np.dot(W_V, h) + B_V
        y_hat = softmax(v)
        target = 0
        m = y_hat[target][0]
        for i, v in enumerate(y_hat):
            if v[0] > m:
                target = i
                m = v[0]
        output = num2one_hot(target)
        return h, c, output

    def forwardProp(self):
        global W_F, B_F, W_C, B_C, W_I, B_I, W_O, B_O, W_V, B_V
        """
        return: self.h: h_next
                self.c: c_next
        """
        self.z = np.row_stack((self.x, self.h_prev))  # Z_SIZE x 1
        # self.z = self.z.asnumpy()        
        self.f = sigmoid(np.dot(W_F, self.z) + B_F)  # H_SIZE x 1
        self.c_bar = np.tanh(np.dot(W_C, self.z) + B_C)  # H_SIZE x 1
        self.o = sigmoid(np.dot(W_O, self.z) + B_O)  # H_SIZE x 1
        self.i = sigmoid(np.dot(W_I, self.z) + B_C)  # H_SIZE x 1
        self.c = self.f * self.c_prev + self.i * self.c_bar  # H_SIZE x 1
        self.h = np.tanh(self.c) * self.o  # H_SIZE x 1
        self.v = np.dot(W_V, self.h) + B_V  # Z_SIZE x 1
        self.y_hat = softmax(self.v)  # Z_SIZE x 1
        return self.h, self.c

    def output(self):
        target = 0
        m = self.y_hat[0][0]
        ll = self.y_hat.__len__()
        for i in range(ll):
            if self.y_hat[i][0] > m:
                target = i
                m = self.y_hat[i][0]
        r = np.zeros_like(self.y_hat)
        r[target][0] = 1
        return r

    def get_loss(self):
        # print(self.y.T, '@@@@@@@@@@@@22')
        # print(np.log(self.y_hat+0.00001), '!!!!!!!!!')
        
        self.loss = -np.dot(self.y.T, np.log(self.y_hat+0.00001))  # 1 x 1
        return self.loss

    def backwardProp(self, next_dh, next_dc):
        global W_F, B_F, W_C, B_C, W_I, B_I, W_O, B_O, W_V, B_V
        """
        return: prev_dh: next_dh
                prev_dc: next_dc
        """
        self.d_v = self.y_hat - self.y  # Z_SIZE x 1
        self.d_w_v = np.dot(self.d_v, self.h.T)  # Z_SIZE x H_SIZE
        self.d_b_v = self.d_v * 1  # Z_SIZE x 1
        self.d_h = np.dot(W_V.T, self.d_v) + next_dh  # H_SIZE x 1
        self.d_o = self.d_h * np.tanh(self.c)  # H_SIZE x 1
        self.d_c = (self.d_h * self.o) * (1-np.tanh(self.c)**2) + next_dc  # H_SIZE x 1
        self.d_b_o = self.d_o * self.o*(1-self.o) * 1  # H_SIZE x 1
        self.d_w_o = np.dot(self.d_o*self.o*(1-self.o), self.z.T)  # H_SIZE, Z_SIZE
        self.d_f = self.d_c * self.c_prev  # H_SIZE x 1
        self.d_i = self.d_c * self.c_bar  # H_SIZE x 1
        self.d_c_bar = self.d_c * self.i  # H_SIZE x 1
        self.d_w_i = np.dot(self.d_i*(self.i*(1-self.i)), self.z.T)  # H_SIZE, Z_SIZE
        self.d_b_i = self.d_i*(self.i*(1-self.i))*1  # H_SIZE x 1
        self.d_w_c = np.dot(self.d_c_bar*self.c_bar*(1-self.c_bar), self.z.T)  # H_SIZE x Z_SIZE
        self.d_b_c = self.d_c_bar*self.c_bar*(1-self.c_bar)*1  # H_SIZE x 1
        self.d_w_f = np.dot(self.d_f*self.f*(1-self.f), self.z.T)  # H_SIZE x Z_SIZE
        self.d_b_f = self.d_f*self.f*(1-self.f)*1  # H_SIZE x 1
        self.d_z = np.dot(W_F.T, self.d_f*self.f*(1-self.f)) + np.dot(W_C.T, self.d_c_bar*self.c_bar*(1-self.c_bar)) + \
            np.dot(W_I.T, self.d_i*self.i*(1-self.i)) + np.dot(W_O.T, self.d_o*self.o*(1-self.o))  # Z_SIZE x 1
        d_prev_h = self.d_z[V_SIZE:, :]  # H_SIZE x 1
        d_prev_c = self.d_c * self.f
        return d_prev_h, d_prev_c

    def gradient(self):
        return self.d_w_f, self.d_b_f, self.d_w_c, self.d_b_c, self.d_w_i, self.d_b_i, self.d_w_o, self.d_b_o, self.d_w_v, self.d_b_v


class RNN():

    def __init__(self, sentence, n):
        global W_F, B_F, W_C, B_C, W_I, B_I, W_O, B_O, W_V, B_V
        self.input = sentence[0:-1]
        self.output = sentence[1:]
        self.n = n
        self.loss = 0.0
        self.h_prev = H_INIT
        self.c_prev = C_INIT
        self.cell_list = list()
        self.d_h_next = H_INIT
        self.d_c_next = C_INIT
        self.w_f_update = np.zeros_like(W_F)
        self.b_f_update = np.zeros_like(B_F)
        self.w_c_update = np.zeros_like(W_C)
        self.b_c_update = np.zeros_like(B_C)
        self.w_i_update = np.zeros_like(W_I)
        self.b_i_update = np.zeros_like(B_I)
        self.w_o_update = np.zeros_like(W_O)
        self.b_o_update = np.zeros_like(B_O)
        self.w_v_update = np.zeros_like(W_V)
        self.b_v_update = np.zeros_like(B_V)
           
    def train(self):
        global W_F, B_F, W_C, B_C, W_I, B_I, W_O, B_O, W_V, B_V
        for i, current_input in enumerate(self.input):
            x = current_input
            y = self.output[i]
            lstm = LSTM(x, y, self.h_prev, self.c_prev)
            self.cell_list.append(lstm)
            self.h_prev, self.c_prev = lstm.forwardProp()
        length = self.cell_list.__len__()
        for j in range(length):
            lstm = self.cell_list[-1-j]
            self.d_h_next, self.d_c_next = lstm.backwardProp(self.d_h_next, self.d_c_next)
            self.loss += lstm.get_loss()
            d_w_f, d_b_f, d_w_c, d_b_c, d_w_i, d_b_i, d_w_o, d_b_o, d_w_v, d_b_v = lstm.gradient()
            self.w_f_update += d_w_f
            self.b_f_update += d_b_f
            self.w_c_update += d_w_c
            self.b_c_update += d_b_c
            self.w_i_update += d_w_i
            self.b_i_update += d_b_i
            self.w_o_update += d_w_o
            self.b_o_update += d_b_o
            self.w_v_update += d_w_v
            self.b_v_update += d_b_v
        if CLIP_GRAD:
            self.clip()
        W_F = optimizer1.optimize(self.n, self.w_f_update, W_F)
        B_F = optimizer2.optimize(self.n, self.b_f_update, B_F)
        W_C = optimizer3.optimize(self.n, self.w_c_update, W_C)
        B_C = optimizer4.optimize(self.n, self.b_c_update, B_C)
        W_I = optimizer5.optimize(self.n, self.w_i_update, W_I)
        B_I = optimizer6.optimize(self.n, self.b_i_update, B_I)
        W_O = optimizer7.optimize(self.n, self.w_o_update, W_O)
        B_O = optimizer8.optimize(self.n, self.b_o_update, B_O)
        W_V = optimizer9.optimize(self.n, self.w_v_update, W_V)
        B_V = optimizer10.optimize(self.n, self.b_v_update, B_V)
         
    def clip(self):
        self.w_f_update = np.clip(self.w_f_update, -clip_value, clip_value)
        self.b_f_update = np.clip(self.b_f_update, -clip_value, clip_value)
        self.w_c_update = np.clip(self.w_c_update, -clip_value, clip_value)
        self.b_c_update = np.clip(self.b_c_update, -clip_value, clip_value)
        self.w_i_update = np.clip(self.w_i_update, -clip_value, clip_value)
        self.b_i_update = np.clip(self.b_i_update, -clip_value, clip_value)
        self.w_o_update = np.clip(self.w_o_update, -clip_value, clip_value)
        self.b_o_update = np.clip(self.b_o_update, -clip_value, clip_value)
        self.w_v_update = np.clip(self.w_v_update, -clip_value, clip_value)
        self.b_v_update = np.clip(self.b_v_update, -clip_value, clip_value)
        
    def get_gradient(self):
        return self.w_f_update, self.b_f_update, self.w_c_update, self.b_c_update, self.w_i_update, self.b_i_update, self.w_o_update, self.b_o_update, self.w_v_update, self.b_v_update

    def get_loss(self):
        return self.loss

    def sample(self, input_word_vector, length):
        input_word = parse_one_hot(input_word_vector)
        word = [input_word]
        next_h = H_INIT
        next_c = C_INIT
        output = input_word_vector
        for i in range(length):
            lstm = LSTM(output, [], h_prev=next_h, c_prev=next_c)
            next_h, next_c, output = lstm.sample()
            word.append(parse_one_hot(output))
        print(''.join([q for q in word]))
        return word


if __name__ == '__main__':
    train()
