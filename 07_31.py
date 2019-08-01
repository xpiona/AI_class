import pandas as pd
import numpy as np

x =np.random.rand(3)
print(x)

# w, b = 2, 3
y = 2*x + 3
print(y)

w = np.random.rand()
b = np.random.rand()
print(w,b)

# alpha = 0.1
# for i in range(0,1000):
#     k = w * x + b
#     error = k - y
#     if(k > y):
#         w = w - w*alpha
#         b = b - b*alpha
#     elif(k < y):
#         w = w + w*alpha
#         b = b + b*alpha
#
#     if( i % 7 ==0):
#         print(error)

n = 1000
lr = 1

for i in range(n):
    y_pred = w * x +b
    e = y_pred - y
    w = w - lr * (e * x).mean()
    b = b - lr * e.mean()
    print('error : ', e)
