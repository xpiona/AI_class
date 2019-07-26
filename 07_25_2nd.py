from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN, LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
import matplotlib.pyplot as plt

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=10000)
np.load = np_load_old

input_train = sequence.pad_sequences(input_train, maxlen=500)
input_test = sequence.pad_sequences(input_test, maxlen=500)

model = Sequential()
model.add(Embedding(10000, 32)) #32차원 32개로 표현
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_acc']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'training_acc')
plt.plot(epochs, val_acc, 'b', label = 'validation_acc')
plt.legend()
plt.show()
