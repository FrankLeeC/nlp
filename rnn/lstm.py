# -*- coding:utf-8 -*-
import numpy as np
import codecs
import collections
import math
import win_unicode_console
win_unicode_console.enable()

PATH = './test.txt'
DATASET = []
with codecs.open(PATH, 'r', encoding='utf-8') as file:
    line = ''
    while True:
        line = file.readline().strip()
        if not line:
            break
        else:
            DATASET.extend(line.split(' '))

H_SIZE = 100
V_SIZE = set(DATASET).__len__()
Z_SIZE = H_SIZE + V_SIZE
WORD2INT = {}
INT2WORD = {}
RATE = 0.01
TRAIN_ITER = 5000
PRINT_ITER = 50

count = []
count.extend(collections.Counter(DATASET).most_common(V_SIZE))
for i, v in enumerate(count):
    WORD2INT[v[0]] = i
INT2WORD = dict(zip(WORD2INT.values(), WORD2INT.keys()))

W_F = 2*np.random.rand(H_SIZE, Z_SIZE)-1
B_F = np.random.rand(H_SIZE, 1)
W_C = 2*np.random.rand(H_SIZE, Z_SIZE)-1
B_C = np.random.rand(H_SIZE, 1)
W_I = 2*np.random.rand(H_SIZE, Z_SIZE)-1
B_I = np.random.rand(H_SIZE, 1)
W_O = 2*np.random.rand(H_SIZE, Z_SIZE)-1
B_O = np.random.rand(H_SIZE, 1)
W_V = 2*np.random.rand(V_SIZE, H_SIZE)-1
B_V = np.random.rand(V_SIZE, 1)
H_INIT = np.zeros((H_SIZE, 1), dtype=float)
C_INIT = np.zeros((H_SIZE, 1), dtype=float)


def word2one_hot(x):
    return num2one_hot(WORD2INT[x])


def num2one_hot(n):
    a = np.zeros((V_SIZE, 1))
    a[n][0] = 1
    return a


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


def sample():
    word = 'so'
    input_word_vector = word2one_hot(word)
    rnn = RNN('', 0)
    rnn.sample(input_word_vector)


def save():
    base = './test/'
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


def train():
    lines = []
    with codecs.open(PATH, 'r', 'utf-8') as file:
        line = ''
        while True:
            line = file.readline().strip()
            if not line:
                break
            lines.append(line)

    n = 0
    c = 0
    for m in range(TRAIN_ITER):
        for i, l in enumerate(lines):
            n = i+1
            c += 1
            raw_sentence = []
            raw_sentence.extend(l.split(' '))
            sentence = []
            for j, v in enumerate(raw_sentence):
                sentence.append(word2one_hot(v))
            rnn = RNN(sentence, n)
            rnn.train()
            loss = rnn.get_loss()
            if c % PRINT_ITER == 0:
                print('%d loss: %f' % (c, loss))
                sample()
                print('-------------')
    save()
    print('save over!')
        

def adam_optimizre(b1, b2, t, g, m, rate, epsilon):
    v1 = 0.0
    c1 = 0.0
    v1 = b1*v1+(1-b1)*g
    c1 = b2*c1+(1-b2)*(g**2)
    v = v1/(1-b1**t)
    c = c1/(1-b2**t)
    print((rate*v/(np.sqrt(c)+epsilon)).shape, m.shape)
    m -= rate*v/(np.sqrt(c)+epsilon)
    return m


class LSTM():
    
    def __init__(self, x, y, h_prev, c_prev):
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        self.y = y


    def sample(self):
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
        """
        return: self.h: h_next
                self.c: c_next
        """
        self.z = np.row_stack((self.x, self.h_prev))  # Z_SIZE x 1
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
        l = self.y_hat.__len__()
        for i in range(l):
            if self.y_hat[i][0] > m:
                target = i
                m = self.y_hat[i][0]
        r = np.zeros_like(self.y_hat)
        r[target][0] = 1
        return r


    def get_loss(self):
        self.loss = -np.dot(self.y.T, np.log(self.y_hat))  # 1 x 1
        return self.loss


    def backwardProp(self, next_dh, next_dc):
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
        self.d_z = np.dot(W_F.T, self.d_f*self.f*(1-self.f)) + np.dot(W_C.T, self.d_c_bar*self.c_bar*(1-self.c_bar)) + np.dot(W_I.T, self.d_i*self.i*(1-self.i)) + np.dot(W_O.T, self.d_o*self.o*(1-self.o))  # Z_SIZE x 1
        d_prev_h = self.d_z[V_SIZE:, :]  # H_SIZE x 1
        d_prev_c = self.d_c * self.f
        return d_prev_h, d_prev_c


    def gradient(self):
        return self.d_w_f, self.d_b_f, self.d_w_c, self.d_b_c, self.d_w_i, self.d_b_i, self.d_w_o, self.d_b_o, self.d_w_v, self.d_b_v
    

class RNN():

    def __init__(self, sentence, n):
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
        adam_optimizre(0.9, 0.99, self.n, self.w_f_update, W_F, RATE, 1e-8)
        adam_optimizre(0.9, 0.99, self.n, self.b_f_update, B_F, RATE, 1e-8)
        adam_optimizre(0.9, 0.99, self.n, self.w_c_update, W_C, RATE, 1e-8)
        adam_optimizre(0.9, 0.99, self.n, self.b_c_update, B_C, RATE, 1e-8)
        adam_optimizre(0.9, 0.99, self.n, self.w_i_update, W_I, RATE, 1e-8)
        adam_optimizre(0.9, 0.99, self.n, self.b_i_update, B_I, RATE, 1e-8)
        adam_optimizre(0.9, 0.99, self.n, self.w_o_update, W_O, RATE, 1e-8)
        adam_optimizre(0.9, 0.99, self.n, self.b_o_update, B_O, RATE, 1e-8)
        adam_optimizre(0.9, 0.99, self.n, self.w_v_update, W_V, RATE, 1e-8)
        adam_optimizre(0.9, 0.99, self.n, self.b_v_update, B_V, RATE, 1e-8)
        

    def get_loss(self):
        return self.loss


    def sample(self, input_word_vector):
        input_word = parse_one_hot(input_word_vector)
        word = [input_word]
        lstm = LSTM(input_word_vector, [], h_prev=H_INIT, c_prev=C_INIT)
        next_h, next_c, output = lstm.sample()
        word.append(parse_one_hot(output))
        lstm = LSTM(output, [], h_prev=next_h, c_prev=next_c)
        next_h, next_c, output = lstm.sample()
        word.append(parse_one_hot(output))
        lstm = LSTM(output, [], h_prev=next_h, c_prev=next_c)
        next_h, next_c, output = lstm.sample()
        word.append(parse_one_hot(output))
        lstm = LSTM(output, [], h_prev=next_h, c_prev=next_c)
        next_h, next_c, output = lstm.sample()
        word.append(parse_one_hot(output))
        print(word)


if __name__ == '__main__':
    train()
