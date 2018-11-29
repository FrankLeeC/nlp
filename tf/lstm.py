# -*-coding:utf-8-*-

import numpy as np
import tensorflow as tf

def num2one_hot(n, size):
    targets = np.array([n]).reshape(-1)
    d = np.eye(size)[targets]
    return d

word = 'long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .'
w2i = {}
i2w = {}
ds = []
data = word.split()
s = set(data)
vsize = s.__len__()
for i, v in enumerate(s):
    w2i[v] = i
    i2w[i] = v
for w in data:
    ds.append(num2one_hot(w2i[w], vsize))
batch_size = 3
step_size = 10
iterator = data.__len__() // (batch_size * step_size)
epochs = 100
hsize=50
ds = np.array(ds).reshape([batch_size, -1, vsize])
with tf.device("/cpu:0"):
    vcabs = tf.Variable(np.eye(vsize))
    embedding = tf.nn.embedding_lookup(vcabs, ds)
init_hidden = tf.placeholder(dtype=tf.float32, shape=[hsize, 1])
init_state = tf.placeholder(dtype=tf.float32, shape=[hsize, 1])
init_tuple = tf.nn.rnn_cell.LSTMStateTuple([init_hidden, init_state])
cell = tf.nn.rnn_cell.BasicLSTMCell(hsize)
output, state = tf.nn.dynamic_rnn(cell, embedding, initial_state=init_tuple)
for i in range(epochs):
    for j in range(iterator):
        s = j * step_size
        e = s + step_size
        word[s: e]
        
        