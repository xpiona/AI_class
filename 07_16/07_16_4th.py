import numpy as np
import matplotlib as plt


def function_1(x):
    return 0.01*x**2 + 0.1 * x


def function_2(x):
    return x[0]**2 + x[1]**2


def numerical_differential_1(f, x):
    h= 10e-5
    return (f(x +h) - f(x)) /h


def numerical_differential_2(f,x):
    h = 1e-1
    grad = np.zeros_like(x)

    for index in range(x.size):
        tmp_val = x[index]
        x[index] = tmp_val + h
        fxh1 = f(x)

        x[index] = tmp_val - h
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2*h)
        x[index] = tmp_val
    return grad



