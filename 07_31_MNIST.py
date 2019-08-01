from keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import tree
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import mape
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from keras.models import Model, Sequential
from keras.layers import Dense,Flatten
from keras import layers
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

X_train1 = X_train.reshape((-1, 28, 28, 1))
X_test1 = X_test.reshape((-1, 28, 28, 1))


X_train2 = X_train.reshape((60000, 1, 784))
X_test2 = X_test.reshape((10000, 1, 784))

X_train_new = X_train.reshape(X_train.shape[0], 784).astype('float64') / 255
X_test_new = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255
Y_train_new = np_utils.to_categorical(Y_train, 10)
Y_test_new = np_utils.to_categorical(Y_test, 10)

print(X_train1.shape)
print(X_train.shape)
print(X_train_new.shape)
print(Y_train.shape)
print(Y_train_new.shape)

# rf = RandomForestClassifier(n_estimators=50)
# rf.fit(X_train_new, Y_train_new)
# print('랜덤 포레스트의 score : ', rf.score(X_test_new, Y_test_new))

# lr = LogisticRegression()
# lr.fit(X_train_new, Y_train_new[1])
# print('로지스틱의 score : ', lr.socre(X_test_new, Y_test))

# model1 = Sequential()
# model1.add(Dense(64, activation='relu', input_shape = (28, 28) ))
# model1.add(Dense(64, activation='relu'))
# model1.add(Flatten())
# model1.add(Dense(10, activation='softmax'))
# model1.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model1.fit(X_train, Y_train_new)
# model1_test_loss, model1_test_acc = model1.evaluate(X_test, Y_test_new)
# print(model1_test_loss, model1_test_acc)

# model4 = Sequential()
# model4.add(Dense(64, activation='relu', input_shape = (1, 784)))
# model4.add(Dense(64, activation='relu'))
# model4.add(Flatten())
# model4.add(Dense(10,activation='softmax'))
# model4.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
# model4.fit(X_train2, Y_train_new)
# model4_test_loss, model4_test_acc = model4.evaluate(X_test2, Y_test_new)
# print(model4_test_loss, model4_test_acc)

# model2 = Sequential()
# model2.add(layers.Conv2D(64, (3,3), activation='relu', input_shape = (28,28,1)))
# model2.add(layers.MaxPooling2D(2,2))
# model2.add(layers.Conv2D(32, (3,3), activation='relu'))
# model2.add(layers.MaxPooling2D(2,2))
# model2.add(Flatten())
# model2.add(Dense(10, activation='softmax'))
# model2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model2.fit(X_train1, Y_train_new)
# model2_test_loss, model2_test_acc = model2.evaluate(X_test1, Y_test_new)
# print(model2_test_loss, model2_test_acc)

# model5 = Sequential()
# model5.add(layers.Conv2D(64, (3,3), activation='relu', input_shape = (1, 784)))
# model5.add(layers.MaxPooling2D(2,2))
# model5.add(layers.Conv2D(32, (3,3), activation='relu'))
# model5.add(layers.MaxPooling2D(2,2))
# model5.add(Flatten())
# model5.add(Dense(10, activation='softmax'))
# model5.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model5.fit(X_train_new, Y_train_new)
# model5_test_loss, model5_test_acc = model5.evaluate(X_test_new, Y_test_new)
# print(model5_test_loss, model5_test_acc)

# print(X_train_new.shape)
# model3 = SGDClassifier(max_iter=500, random_state=31)
# model3.fit(X_train_new, Y_train)
# print('SGD의 score : ', model3.score(X_test_new, Y_test))
