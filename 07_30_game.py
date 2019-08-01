import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

pass_time = [8, 9, 9,   9.5, 10,  12,  14,  14.5, 15,  16,  16,  16.5, 17, 17, 17,  17.5,  20, 20, 20]
fail_time = [1, 2, 2.1, 2.6, 2.7, 2.8, 2.9, 3,    3.2, 3.4, 3.5, 3.6,  3,  5,  5.2, 5.4  , 3.3, 2.1, 5.2]

data = {
    "time" : [8, 9, 9,   9.5, 10,  12,  14,  14.5, 15,  16,  16,  16.5, 17, 17, 17,  17.5,  20, 20, 20,1, 2, 2.1, 2.6, 2.7, 2.8, 2.9, 3,    3.2, 3.4, 3.5, 3.6,  3,  5,  5.2, 5.4  , 3.3, 2.1, 5.2],
    "game_time" : [1, 3, 2,   4,    2,   1,   2,   3,    1,   3,   3,   2,    1,  1,  1,   2,    1,  1 ,  1, 5, 6, 4, 5, 6, 5, 4, 6, 5, 6, 4, 5, 6, 5, 4, 6, 4,5,   5],
}
y1=[1]*len(pass_time)
y0=[0]*len(fail_time)
y = np.hstack((y1,y0))
r = pd.DataFrame(data)
r['result'] =  y

X = pd.DataFrame(list(zip(data['time'], data['game_time'])))
y = r['result']
print(len(X), len(y))

fig = plt.figure(figsize=(8,6))
plt.xlim(0, 21)
plt.ylim(-0.1, 1.1)
plt.xlabel("Study time")
plt.ylabel("Pass rate")
plt.scatter(X[0], y)
plt.scatter(X[1], y, c = 'r')
plt.show()



# model = LogisticRegression()
# model.fit(X.reshape(-1, 1),y)
#
# print(model.coef_)
# print(model.intercept_)
#
# def logreg(z):
#     return 1 /(1+np.exp(-z))
#
# fig = plt.figure(figsize=(8,6))
# plt.xlim(0, 20)
# plt.ylim(-0.1, 1.1)
# plt.xlabel('Study time')
# plt.ylabel('Pass rate')
# plt.scatter(X, y, s=50)
#
# XX = np.linspace(0.5,21, 100)
# yy = logreg(model.coef_*XX + model.intercept_)[0]
# plt.plot(XX, yy, c='r')
