import keras
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers

# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=20000)

# restore np.load for future normal usage
np.load = np_load_old

max([max(sequence) for sequence in train_data])

def vectorize_sequences(sequence, dimension = 20000):
    results = np.zeros((len(sequence), dimension))
    for i,sequence in enumerate(sequence):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = models.Sequential()
model.add(layers.Dense(128, activation='tanh', input_shape = (20000,)))
# model.add(layers.Dense(32, activation='tanh'))
# model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=  keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                loss='binary_crossentropy',
                metrics=['accuracy'])

# history = model.fit(x_train, y_train, epochs=20, batch_size=512)
# model.fit(x_train, y_train, epochs=20, batch_size=512)
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print('result = ', results)
