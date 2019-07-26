import keras
from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old

word_index = reuters.get_word_index()
reverse = dict([(value, key) for (key, value) in word_index.items()])
decoded =  ' '.join([reverse.get(i-3, '#') for i in train_data[0]])

def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension = 46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


y_train = to_one_hot(train_labels)
y_test  = to_one_hot(test_labels)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer ='Adam', loss = 'categorical_crossentropy',
                metrics=['accuracy'])

x_val =x_train[:2000]
partial_x_train = x_train[2000:]

y_val = y_train[:2000]
partial_y_train = y_train[2000:]

# history = model.fit(partial_x_train, partial_y_train,
                    # epochs=12, batch_size=512,
                    # validation_data=(x_val, y_val))

history = model.fit(partial_x_train, partial_y_train,
                    epochs=12, batch_size=512)

print(model.evaluate(x_test, y_test))
predictions = model.predict(x_test)
