from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import sys

import io



sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

base_dir ='C:\\Users\\jaehoon\\Desktop\\deep-learning-with-python-notebooks-master\\datasets\\cats_and_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


def extract_feature(directory, sample_count):
  features = np.zeros(shape=(sample_count, 4, 4, 512))
  labels = np.zeros(shape=(sample_count))
  generator = datagen.flow_from_directory(directory,
                                         target_size=(150,150),
                                         batch_size=batch_size,
                                         class_mode='binary')
  i=0
  for inputs, labels_batch in generator:
    features_batch = conv_base.predict(inputs)
    features[i * batch_size : (i+1) * batch_size] = features_batch
    labels[i * batch_size : (i+1) * batch_size] = labels_batch
    i +=1
    if i * batch_size >= sample_count:
      break
  return features, labels

datagen = ImageDataGenerator(rescale=1./255)

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1,activation ='sigmoid'))

print(model.summary)

len(model.trainable_weights)
len(conv_base.trainable_weights)

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2, horizontal_flip=True,
                                    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                    target_size=(150, 150),
                                    batch_size=20,
                                    class_mode = 'binary')


validation_generator = train_datagen.flow_from_directory(validation_dir,
                                    target_size=(150, 150),
                                    batch_size=20,
                                    class_mode = 'binary')

model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(lr = 2e-5),
                metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=100,
                                epochs=30, validation_data=validation_generator,
                                validation_steps=50)
