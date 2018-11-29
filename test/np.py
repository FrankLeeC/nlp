# -*- coding:utf-8 -*-

import numpy as np
import win_unicode_console as fuckkkkkkkkkkkkkyou_windows
fuckkkkkkkkkkkkkyou_windows.enable()

a = np.random.rand(3, 1)
b = np.random.rand(5, 1)
print(a)
print(b)
print(a.shape, b.shape)
# c = [a, b]
# print(c)
# print(c.shape)

d = np.row_stack((a, b))
print(d)
print(d.shape)
print('-----------')

c = np.random.rand(3, 1)
print(c)
print(np.max(c))
print(np.argmax(c))

print('----------------------')
a = np.random.randint(0, 10, (5, 5))
print(a)
# a.itemset((0, 2), (1,2), (3, 4), 12)
# print(a)
print('----------------')
a = np.random.randint(0, 10, (5, 1))
print(a)
b = a.take((0, 2, 4))
print(np.asmatrix(b).T)
print('-------------------')
hex = int('0x10', 16)
print(hex)

print('----------')
# print(np.random.rand(3, 2))
print(np.random.random(size=(3, 2)))