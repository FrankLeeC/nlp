# -*- coding: utf-8 -*-

import numpy as np

class Adagrad:

    def __init__(self, init_n=0, learning_rate=0.1, epsilon=0.00001):
        self.prev_n = init_n
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def get_grad(self, grad):
        n = self.prev_n + grad*grad
        self.prev_n = n
        return -self.learning_rate*grad/(np.sqrt(n+self.epsilon))


class Adam:
    
    def __init__(self, b1=0.9, b2=0.999, learning_rate=0.1, epsilon=0.00001):
        self.v1 = 0
        self.c1 = 0
        self.b1 = b1
        self.b2 = b2
        self.rate = learning_rate
        self.epsilon = epsilon
        self.t = 1

    def get_grad(self, grad):
        self.v1 = self.b1*self.v1+(1-self.b1)*grad
        self.c1 = self.b2*self.c1+(1-self.b2)*(grad**2)
        v = self.v1/(1-self.b1**self.t)
        c = self.c1/(1-self.b2**self.t)
        self.t += 1
        return -self.rate*v/(np.sqrt(c)+self.epsilon)


def create_optimizer(name='Adagrad'):
    if name == 'Adagrad':
        return Adagrad()
    elif name == 'Adam':
        return Adam()