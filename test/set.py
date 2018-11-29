# -*-coding:utf-8-*-
import numpy as np
import math
import win_unicode_console
win_unicode_console.enable()
s = set()
p = '寒随穷律变春逐鸟声开初风飘带柳晚雪间花梅碧林青旧竹绿沼翠新苔芝田初雁去绮树巧莺来'
for i in p:
    if s.__contains__(i):
        print(i)
    s.add(i)
print(s.__len__())

a = np.mat(np.random.random((3, 2)))
print(a)
print('-------------------')
b = np.array([1, 2, 3, 4, 5])

c = np.random.normal(0, 0.1, 15)
d = np.mat(c).reshape([3, 5])
print(d)
np.put(d, [5, 6, 7, 8, 9], b)
print('-------------')
print(d)

print('----------------')
print(math.log(0.0000000000001))

print('----------')
print(np.random.randn(5, 3))

print(10 // 3)

print('===================================')

def softmax(a):
    m = np.max(a)
    print('max', m)
    print(a-m)
    s = np.sum(np.exp(a - m))
    print('s:', s)    
    return np.exp(a - m) / s


# print(softmax(np.array([[-0.17490368],[ 0.68224721],[-1.07185286],[ 0.21533988],[ 0.18988101],
# [-1.28023877],[-0.29321016],[ 0.64675704],[-0.88588487],[ 1.83709917],[ 1.30324512],[-0.59002982],
# [ 0.70091933],[ 0.54996511],[ 0.59918394],[ 0.18284976],[-0.95051301],[ 0.69385861],[ 0.5467939 ],[ 0.57296248],[ 1.15643409],[-1.50956556],[ 0.19294956],[ 0.26613252]])))


a = [10, 10, 9, 5, 7, 8]
print(a[0:10])
