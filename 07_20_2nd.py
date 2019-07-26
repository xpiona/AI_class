from keras.datasets import boston_housing
import keras
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras import backend as K
import matplotlib.pyplot as plt
import sys
import io
import tensorflow as tf

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

# restore np.load for future normal usage
np.load = np_load_old

mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
train_data = (train_data - mean) / std

test_data -= mean
test_data /= std



def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='Adam', loss = 'mse', metrics=['mae'])
    #mae는 단위르 기준으로 얼마나 틀렸는지
    return model

def smooth_scurve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_ponts:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

k=5
num_val_samples = len(train_data) //k
num_epochs = 500
all_scores = []
all_mae = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_target[i* num_val_samples : (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis = 0)
    partial_train_target = np.concatenate(
        [train_target[:i * num_val_samples],
        train_target[(i + 1) * num_val_samples:]],
        axis = 0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_target,
                epochs=num_epochs, batch_size=1
                )


    mae_history = history.history['mean_absolute_error']
    all_mae.append(mae_history)

    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    # all_scores.append(val_mae)

smooth_mae_history = smooth_scurve(average_mae_history[10:])
plt.plt(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epoche')
plt.ylabel('Validation MAE')
average_mae_history = [np.mean([x[i] for x in all_mae]) for i in range(num_epochs) ]
plt.plot(range(1, len(average_mae_history)))
plt.show()
