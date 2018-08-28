# -*- coding:utf8 -*-

import codecs
import collections
import random
import time
import numpy as np
import math
import win_unicode_console as fuck_windows
import copy
import matplotlib
import matplotlib.pyplot as plt
fuck_windows.enable()

qaz = '-'*10
rate = 0.01
poetry_list = []

look_up_table = {}  # word -> num
reverse_look_up_table = {}  # num -> word

word_vector = []
context_matrix = []  # N x embedding_size
embedding_size = 2
one_hot_list = []


def init_vector():
    global word_vector
    for i in range(look_up_table.__len__()):
        word_vector.append(2*np.random.random([1, embedding_size])-1)  # [-1, 1)
    

def normalize_row(a):
    n = a.shape[0]
    a = a / np.sqrt(np.sum(a**2)).reshape(n, 1)
    return a


def softmax(a):
    m = np.max(a)
    s = np.sum(np.exp(a - m))
    return np.exp(a - m) / s


def init_context():
    global context_matrix
    context_matrix = np.random.randn(look_up_table.__len__(), embedding_size)


def init_ont_hot():
    global one_hot_list
    for i in range(look_up_table.__len__()):
        tmp = np.zeros([1, look_up_table.__len__()])
        tmp[0][i] = 1
        one_hot_list.append(tmp)


def init_look_up_table(corpus):
    global look_up_table, reverse_look_up_table
    count = []
    count.extend(collections.Counter(corpus).most_common(corpus.__len__()))
    for w, c in count:
        look_up_table[w] = look_up_table.__len__()
    reverse_look_up_table = dict(zip(look_up_table.values(), look_up_table.keys()))
        

class Poetry:

    def __init__(self):
        self.length = 0
        self.data = ''
        self.data_set = []
        self.reversed_dictionary = {}
        self.batch = None
        self.labels = None
        self.data_index = 0


    def set_data(self, data_list):
        self.data = data_list


    def build(self):
        for d in self.data:
            self.data_set.append(look_up_table[d])
        self.length = self.data_set.__len__()


    def generate_batch(self, num_skip, skip_window):
        if num_skip <= skip_window * 2:
            span = 2 * skip_window + 1
            buffer = collections.deque(maxlen=span)
            if self.data_index >= self.data_set.__len__():
                self.data_index = 0
            buffer.extend(self.data_set[self.data_index: self.data_index + span])
            self.data_index += span
            batch_size = (self.length - 1) * num_skip
            self.batch = np.ndarray(shape=batch_size, dtype=np.int32)
            self.labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
            for i in range(batch_size // num_skip):
                context_word = [w for w in range(span) if w != skip_window]
                word_to_ues = random.sample(context_word, span - 1)
                for j, w in enumerate(word_to_ues):
                    self.batch[i * num_skip + j] = buffer[skip_window]
                    self.labels[i * num_skip + j, 0] = buffer[w]
                if self.data_index >= self.data_set.__len__():
                    buffer = self.data_set[:span]
                    self.data_index = span
                else:
                    buffer.append(self.data_set[self.data_index])
                    self.data_index += 1
            self.data_index = (self.data_index + len(self.data_set) - span) % len(self.data_set)
            return self.batch, self.labels


def read():
    global poetry_list
    with codecs.open('./test.txt', mode='r', encoding='utf-8') as file:
        while True:
            line = file.readline().strip()
            if line:
                line = line.replace('：', ':').replace(' ', '').replace('，', ',').replace('。', '.')
                tmp = line.split(":", 1)
                if tmp.__len__() > 1 and tmp[1].__contains__(','):
                    p = Poetry()
                    p.set_data(tmp[1].replace(',', '').replace('.', ''))
                    poetry_list.append(p)
            else:
                break


def draw(b):
    fig, ax = plt.subplots()
    for i, v in enumerate(word_vector):
        ax.plot(v[0][0], v[0][1], 'o')
        ax.text(v[0][0], v[0][1], reverse_look_up_table[i], family='ZHCN', ha='right', wrap=True)
    plt.show(block=b)

def init(s):
    init_look_up_table(s)
    init_vector()
    init_context()
    init_ont_hot() 

if __name__ == '__main__':
    tick = time.time()
    read()
    s = ''.join([p.data for p in poetry_list])
    init(s)
    draw(False)
    
    one_vector = np.ones([look_up_table.__len__(), 1])
    cal_count = 0
    num_skip = 2
    skip_window = 1
    for poem in poetry_list:
        poem.build()
        batch, labels = poem.generate_batch(num_skip, skip_window)
        print(batch)
        print(labels)
        print('--')
        for num in range(200):
            i = 0
            while i < batch.__len__():
                v = batch[i]
                cal_count += 1
                v_center = word_vector[v]  # 1 x embedding_size
                output = context_matrix.dot(v_center.T)  # N x 1
                prop = softmax(output)  # N x 1

                context_indecies = labels[i:i+num_skip, :]
                cost = 0.0
                center_grad = np.zeros_like(word_vector)  # 1 x embedding_size
                context_grad = np.zeros_like(context_matrix)  # N x  embedding_size
                for context_index in context_indecies:
                    grad_prop = copy.deepcopy(prop)
                    cost += -math.log(grad_prop[context_index]+0.00001)
                    grad_prop[context_index] -= 1
                    center_grad[v] = center_grad[v] + grad_prop.T.dot(context_matrix)  # 1 x embedding_size
                    context_grad = context_grad + grad_prop.dot(v_center)  # N x embedding_size
                
                word_vector -= rate*center_grad
                context_matrix -= rate*context_grad
                
                i += num_skip

                if cal_count % 100 == 0:
                    print('cost:', cost)
                    print(qaz, str(cal_count), qaz)

    tock = time.time()
    cost = tock - tick
    print(qaz, 'cost: ', cost, 's', qaz)
    print(qaz, 'over', qaz)
    print('---')
    draw(True)
