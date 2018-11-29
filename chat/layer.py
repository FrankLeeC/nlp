# -*- coding: utf-8 -*-

import numpy as np
import optimizer as opt
import function as fn


class WordEmbeddingLayer:

    def __init__(self, shape):
        """
        shape: [vsize, embedding_size]
        """
        self.w = np.random.random(shape) - 0.5
        self.g = opt.create_optimizer()

    def call(self, v):
        """
        embedding lookup
        v: one-hot vector
        """
        return np.dot(self.w, v)

    def update(self, grad):
        self.w += self.g.get_grad(grad)
        
        
class SoftmaxLayer:

    def __init__(self, shape):
        """
        shape: [osize, hsize]
        """
        self.w = np.random.random(shape) - 0.5
        self.b = np.random.random((shape[0], 1)) - 0.5
        self.wg = opt.create_optimizer()
        self.bg = opt.create_optimizer()
    
    def call(self, v):
        return fn.softmax(np.dot(self.w, v) + self.b)

    def update(self, w_grad, b_grad):
        self.w += self.wg.get_grad(w_grad)
        self.b += self.bg.get_grad(b_grad)


class EncoderLayer:

    def __init__(self, batch_size, cell, is_output=False):
        self.cell = cell
        self.h_prev = 0
        self.iter = 0
        self.batch_size = batch_size
        self.is_output = is_output
        pass

    def call(self, x):
        r, z, hs, h = self.cell.step(x, self.h_prev)
        self.h_prev = h
        self.cell.store(r, z, hs, h)
        return h

    def backward(self, err):
        self.iter += 1
        rs = None
        if self.is_output:  # err is vector
            rs = self.cell.backward(0, err)
        else:  # err is list of vector
            rs = self.cell.backward(err, 0)
        if self.iter % self.batch_size == 0:
            self.cell.update(self.batch_size)
        return rs


class DecoderLayer:

    def __init__(self, batch_size, cell, h_prev, c, is_output=False):
        self.batch_size = batch_size
        self.h_prev = h_prev
        self.iter = 0
        self.cell = cell
        self.is_output = is_output

    def call(self, x):
        r, z, hs, h = self.cell.step(x, self.h_prev)
        self.h_prev = h
        self.cell.store(r, z, hs, h)
        return h

    def backward(self, err):
        self.iter += 1
        rs = None
        if self.is_output:  # err is vector
            rs = self.cell.backward(0, err)
        else:  # err is list of vector
            rs = self.cell.backward(err, 0)
        if self.iter % self.batch_size == 0:
            self.cell.update(self.batch_size)
        return rs



class DropoutLayer:

    def __init__(self, p):
        self.p = p

    def step(self, x, training=True):
        if training:
            self.u = np.random.binomial(1, self.p, x.shape())
        else:
            self.u = 0.5
        return x*self.u

    def backward(self, err):
        return err * self.u