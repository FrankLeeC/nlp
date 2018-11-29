import minpy.numpy as np
import minpy.numpy.random as random
from minpy.context import set_context, gpu
import win_unicode_console
win_unicode_console.enable()
set_context(gpu(0))

a = random.rand(3, 1)
b = random.rand(5, 1)
print(type(a))
print(type(a.asnumpy()))
c = np.row_stack((a.asnumpy(), b.asnumpy()))
# c = np.concat((a, b), axis=0)
print(c)

# values = np.zeros((1, 10))
# print(values)
# n_values = np.max(values) + 1
# print(n_values)
# d = np.eye(int(n_values[0]))[values]
# print(d)


# nb_classes = 6
# targets = np.array([3, 2]).reshape(-1)
# d = np.eye(nb_classes)[targets]
# print(d.T)

# nb_classes = 6
# targets = np.array([3]).reshape(-1)
# d = np.eye(nb_classes)[targets]
# print(d.T)
print('--------------')
a = 2*random.rand(3, 1)-1
print(a)
b = np.clip(a, -0.001, 0.001)
print(b)