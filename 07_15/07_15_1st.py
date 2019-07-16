import numpy as np

x = np.array([1,2,3])
w= np.array([2,4,6])
b =1.5

tmp = np.sum(x*w) -b

if tmp <= 0:
    print(0)
else:
    print(1)