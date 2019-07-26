from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


embedinng_layer = Embedding(10000, 64)
maxlen= 100
training_samples = 200
validation_samples = 10000
max_words = 10000

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=10000)
np.load = np_load_old

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen = 20)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen = 20)

model = Sequential()
model.add(Embedding(10000, 16, input_length=20))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss ='binary_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

glove_dir = 'C:\\Users\\jaehoon\\Desktop\\glove.6B'

embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coef = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coef
f.close()

print(len(embedding_index))
texts =[]
embedding_dim = 100
embedding_matrix = np.zeros((10000, 100))

tokenizer =Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=maxlen)


for word, i in word_index.items():
    if i < 10000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not null:
            embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(10000, 100, input_length=20))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layer[0].set_weight([embedding_matrix])
model.layer[0].trainable = False
model.compile(optimizer='Adam', loss = 'binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(x_train,y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

model.save_weights('pre_trained_glove.h5')
