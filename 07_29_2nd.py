 # 17,std 3, 100명
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier


mu = 170
std = 2
n = 1000

x = np.random.randn(n)*std + mu
y = x * 1.0 - 105 + np.random.randn(n) * 0.5
m = np.random.randn(n)*std + 160
n = x * 0.7 - 60 + np.random.randn(n) * 0.5

sex = n*[0] +n*[1]
X = pd.DataFrame([x,sex]).T

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=42)
train_m, test_m, train_n, test_n = train_test_split(m, n, test_size=0.33, random_state=42)
train_p, test_p, train_q, test_q = train_test_split(np.concatenate((x,m),axis=0).reshape(-1,1), np.concatenate((y,n)), test_size=0.33, random_state=42)

print(train_x.shape, train_y.shape)
print(train_m.shape, train_n.shape)

plt.scatter(x, y)
plt.scatter(m, n)
plt.scatter(np.concatenate((x,m),axis=0), np.concatenate((y,n),axis=0))
model1 = LinearRegression()
model2 = RandomForestRegressor()
model3 = LinearRegression()
model4 = LinearRegression()
model5 = SGDClassifier()

model1.fit(train_x.reshape(-1, 1), train_y)
model2.fit(train_x.reshape(-1, 1), train_y)
model3.fit(train_m.reshape(-1, 1), train_n)
model4.fit(train_p.reshape(-1, 1), train_q)

a = model1.coef_
b = model1.intercept_
c = model3.coef_
d = model3.intercept_
t = model4.coef_
r = model4.intercept_

xx = np.linspace(160, 180, 3)
yy = a*xx + b
mm = np.linspace(150, 170, 3)
nn = c*mm + d
tt = np.linspace(150,180, 3)
rr = t*tt + r

plt.plot(xx, yy, c='r')
plt.plot(mm, nn, c='b')
plt.plot(tt, rr, c='g')

print('m1_w : ', model1.coef_, 'm1_b : ', model1.intercept_)
print(model2.feature_importances_)
print('m3_w : ', model3.coef_, 'm3_b : ', model3.intercept_)
print('m4_w : ', model4.coef_, 'm4_b : ', model4.intercept_)

print('m1_score : ', model1.score(test_x.reshape(-1, 1), test_y))
print('m2_score : ', model2.score(test_x.reshape(-1, 1), test_y))
print('m3_score : ', model3.score(test_m.reshape(-1, 1), test_n))
print('m4_score : ', model4.score(test_p.reshape(-1, 1), test_q))

plt.show()
