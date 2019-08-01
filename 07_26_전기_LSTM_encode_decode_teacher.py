from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import mape
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from keras.models import Model, Sequential
from keras.layers import GRU,Dense,LSTM,RepeatVector,TimeDistributed,Flatten,Input
from utils import TimeSeriesTensor
from utils import create_evaluation_df

energy = pd.read_csv("energy.csv", parse_dates=["timestamp"], index_col='timestamp')

validation_start_date = '2014-09-01 00:00:00'
test_start_date = '2014-11-01 00:00:00'

# energy[(energy.index < validation_start_date)][['load']].rename(columns={'load':'train'}).join(energy[energy.index >= validation_start_date] & (energy.index < test_start_date)[['load']].rename(columns={'load':'validation'}), how='outer').join(energy[test_start_date:][['load']].rename(columns = {'load':'test'}), how = 'outer').plot()

energy[(energy.index < validation_start_date)][['load']].rename(columns={'load':'train'}).join(energy[(energy.index>=validation_start_date) & (energy.index<test_start_date)][['load']].rename(columns={'load':'validation'}), how='outer')\
.join((energy[test_start_date:][['load']]).rename(columns={'load':'test'}),how='outer').plot()

T = 6
HORIZON = 1

#1단계 데이터나누기
train = energy.copy()[energy.index < validation_start_date][['load', 'temp']]
print(train.head())

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

y_scaler.fit(train[['load']])

train[['load', 'temp']] = x_scaler.fit_transform(train)

print(train.head())



tensor_structure = {'encoder_input': (range(-T+1, 1), ['load', 'temp']), 'decoder_input': (range(0, HORIZON), ['load', 'temp'])}
train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)












# train_shifted =  train.copy()
# train_shifted['y_t+1'] = train_shifted['load'].shift(-1, freq = 'H')
# # print(train_shifted.head())
#
# for t in range(1, T+1):
#     train_shifted[str(T-t)] = train_shifted['load'].shift(T-t, freq = 'H')
#
# X_cols = ['load_t-5', 'load_t-4', 'load_t-3', 'load_t-2', 'load_t-1', 'load_t']
# y_cols = 'y_t+1'
# train_shifted.columns = ['original']+[y_cols]+X_cols
# # print(train_shifted.head())
#
# #결측값 처리
# train_shifted = train_shifted.dropna(how='any')
#
# #5단계 넘파이 어레이배열 (샘플,피쳐)로 바꾸기
# X_train = train_shifted[X_cols].as_matrix()
# y_train = train_shifted[y_cols].as_matrix()











#6단계 검증데이터세트 준비
look_back_date = dt.datetime.strptime(validation_start_date, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)

# print(look_back_date)
valid = energy.copy()[(energy.index >= look_back_date) & (energy.index < test_start_date)][['load', 'temp']]
# print(valid.shape)

#Validation데이터 스케일링
valid[['load','temp']] = x_scaler.transform(valid)
valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)


# valid_shifted = valid.copy()
# valid_shifted['y+1'] = valid_shifted['load'].shift(-1, freq='H')
#
#
# for t in range(1, T+1):
#   valid_shifted['load_t-' + str(T-t)] = valid_shifted['load'].shift(T-t, freq='H')
# valid_shifted.head()
#
#
# valid_shifted = valid_shifted.dropna(how = 'any')
# y_valid = valid_shifted['y+1'].as_matrix()
#
#
# X_valid = valid_shifted[['load_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()
#
unit_dim = 5

bach_size = 32
epochs =5
#
#


encoder_input = Input(shape=(None, 1))
encoder = GRU(unit_dim, return_state=True)
encoder_output, state_h = encoder(encoder_input)
encoder_state = [state_h]

decoder_input = Input(shape=(None, 1))
decoder_GRU = GRU(unit_dim, return_state=True, return_sequences=True)
decoder_output, _ = decoder_GRU(decoder_input, initial_state=encoder_state)
decoder_dense = TimeDistributed(Dense(1))
decoder_output = decoder_dense(decoder_output)

model = Model([encoder_input, decoder_input], decoder_output)

#
#
#
# model = Sequential()
#
# model.add(LSTM(unit_dim1, input_shape=(T,2), return_sequences=True ))
# model.add(LSTM(unit_dim2, input_shape=(T,2)))
# #repeat
# model.add(RepeatVector(3))
#
# model.add(LSTM(unit_dim2,input_shape=(6,2), return_sequences=True))
# model.add(TimeDistributed(Dense(1)))
# model.add(Flatten())
#
# model.compile(optimizer='RMSprop', loss='mse')
# model.summary()
#
earlystop = EarlyStopping(monitor='val_loss', patience=5)
#
# best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)

train_target = train_inputs['target'].reshape(train_inputs['target'].shape[0], train_inputs['target'].shape[1], 1)
valid_target = valid_inputs['target'].reshape(valid_inputs['target'].shape[0], valid_inputs['target'].shape[1], 1)

history = model.fit([train_inputs['encoder_input'], train_inputs['decoder_input']],
                    train_target,
                   batch_size=bach_size,
                   epochs=epochs,
                   validation_data=([valid_inputs['encoder_input'], valid_inputs['decoder_input']], valid_target),
                   callbacks=[earlystop])

#
#
# # best_epoch = np.argmin(np.array(history.history['val_loss'])) +1
# # model.load_weights("model_{:02d}.h5".format(best_epoch))
# #
# plot_df = pd.DataFrame.from_dict({'train_loss':history.history['loss'], 'val_loss':history.history['val_loss']})
# plot_df.plot(figsize=(10,10))
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
# #
# #
# # #########################################################################################################################
# look_back_date = dt.datetime.strptime(test_start_date, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
# look_back_date
#
# test = energy.copy()[look_back_date:][['load', 'temp']]
# test[['load','temp']] = x_scaler.transform(test)
# test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)
# test = energy.copy()[look_back_date:][['load']]
# test['load'] = scaler.transform(test)
# test_shifted = test.copy()
# test_shifted['y+1'] = test_shifted['load'].shift(-1, freq='H')
# test_shifted.head()
# for t in range(1, T+1):
#   test_shifted['load_t-' + str(T-t)] = test_shifted['load'].shift(T-t, freq='H')
# test_shifted
# test_shifted = test_shifted.dropna(how='any')
# y_test = test_shifted['y+1'].as_matrix()
# X_test = test_shifted[['load_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()
#
#
# predictions = model.predict(test_inputs['X'])
# eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, y_scaler)

# eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
# eval_df['timestamp'] = test_shifted.index
# eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
# eval_df['actual'] = np.array(np.transpose(y_test)).ravel()
# eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
# eval_df.head()
# ###################################################################################################################
#
# mape(eval_df['prediction'], eval_df['actual'])
# print(mape(eval_df['prediction'], eval_df['actual']))
#
# eval_df[eval_df.timestamp < '2014-11-08'].plot(x = 'timestamp', y = ['prediction', 'actual'], style = ['r', 'b'], figsize = (15,8))
# plt.show()
