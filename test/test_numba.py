from numba import cuda
import numpy as np
import datetime

print(cuda.get_current_device())

@cuda.jit
def gpumul(a, b, c):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    c[0, idx] = a[0, idx] * b[0, idx]

if __name__ == '__main__':
    a = np.random.randint(0, 10, (1, 300000), dtype=np.int8)
    b = np.random.randint(0, 10, (1, 300000), dtype=np.int8)
    t1 = datetime.datetime.now()
    c1 = a*b
    t2 = datetime.datetime.now()
    print('cpu:', t2-t1)
    c2 = np.zeros_like(a, dtype=np.int8)
    t3 = datetime.datetime.now()
    gpumul[3000, 100](a, b, c2)
    t4 = datetime.datetime.now()
    print('gpu:', t4-t3)
    print(c2-c1)
    # print(c1)
    # print('--------')
    # print(c2)
    # print('----------')
    # print(c1 - c2)