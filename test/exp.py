# -*- coding=utf-8 -*-

import numpy as np

a = np.random.random([10, 1])
print(a)

b = np.exp(a)
print(b / np.sum(b))
