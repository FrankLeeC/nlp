from __future__ import division
import numpy as np
import math
import win_unicode_console
win_unicode_console.enable()

# simple function
# minimum is [0,0,0]
def function(x):
    return x**2

# gradient for the function
def gradient(x):
    return 2*x

# set epsilon in order to avoid division by zero
epsilon = 10**-8

# b1, b2 in [0,1) as exponential decay rates
# alpha is the learning rate
# x is the parameter vector
def adam(alpha, b1, b2, x):
    #initalize m, v, t to zero
    m = 0
    v = 0
    t = 0
    # initialize sentinel to False
    converged = False
    # while the optimization algorithm has not yet converged
    while (not converged):
        # time step increment by 1
        t = t + 1
        # determine the gradient with the current parameter
        g = gradient(x)
        # calculate m with b1
        m = b1*m + ((1 - b1)*g)
        # calculate v with b2
        v = b2*v + ((1 - b2)*(g**2))
        # calculate m_hat with m
        m_hat = m/(1 - b1**t)
        # calculate v_hat with v
        v_hat = v/(1 - b2**t)
        # determine new parameter x_new
        new_x = x - (alpha*m_hat/((v_hat**(1/2)) + epsilon))
        # find the norm of the difference between new_x and x, used as convergence measure
        if np.linalg.norm(new_x - x) <= 0.001:
            converged = True
        # set x to the new_x
        x = new_x
        print(x)


if __name__ == '__main__':
    # initial parameters of a
    a = np.array([100,76,130])
    adam(0.1, 0.9, 0.999, a)