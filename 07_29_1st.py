import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
 # 리스트, 어레이, 시리즈, 튜플, 데이터프레임
li = [1, 2, 3, 4, 5]
li.append(6)
# print(li)

sr = Series([1, 2, 3, 4, 5])

ar = np.array([1, 2, 3, 4, 5])

tp = (1, 2, 3, 4, 5)
# print(type(tp))

df = DataFrame(sr)
# print(df)

n = 6
sum = np.random.randint(1, 7, 1000)
for i in range(0,n):
    sum += np.random.randint(1,7,1000)

plt.hist(np.array(sum), bins = 5*n+1, width = 0.5)
plt.show()
