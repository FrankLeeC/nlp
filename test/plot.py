# -*- coding:utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def function(x):
    return 5*x


def function2(x1, x2):
    return x1 + 2 * x2


a = np.arange(-5, 5, 0.1)    
b = sigmoid(a)
plt.plot(a, b)
plt.show(block=False)



x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
x, y = np.meshgrid(x, y)
z = sigmoid(sigmoid(function2(x, y)))
fig = plt.figure()  
ax = Axes3D(fig)
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')  
plt.show(block=True) 