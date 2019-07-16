import numpy as np

x = np.array([1, 2])
print('shape of imput =', x.shape)
weight = np.array([[1, 3, 5], [2, 4, 6]])
print('weight = ', weight)
print('shape of weight = ', weight.shape)
y = np.dot(x, weight)
print('y = ', y)
