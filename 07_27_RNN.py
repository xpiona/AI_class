
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import mape
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from keras.models import Model, Sequential
from keras.layers import GRU,Dense

energy = pd.read_csv("energy.csv", parse_dates=["timestamp"], index_col='timestamp')
# print(energy.head())

validation_start_date = '2014-09-01 00:00:00'
test_start_date = '2014-11-01 00:00:00'

# energy[(energy.index < validation_start_date)][['load']].rename(columns={'load':'train'}).join(energy[energy.index >= validation_start_date] & (energy.index < test_start_date)[['load']].rename(columns={'load':'validation'}), how='outer').join(energy[test_start_date:][['load']].rename(columns = {'load':'test'}), how = 'outer').plot()
energy[(energy.index < validation_start_date)][['load']].rename(columns={'load':'train'}).join(energy[(energy.index>=validation_start_date) & (energy.index<test_start_date)][['load']].rename(columns={'load':'validation'}), how='outer')\
.join((energy[test_start_date:][['load']]).rename(columns={'load':'test'}),how='outer').plot()

T = 6
HORIZON = 1

train = energy.copy()[energy.index < validation_start_date][['load']]
# print(train.head())

scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)
# print(train.head())

train_shifted =  train.copy()
train_shifted['y_t+1'] = train_shifted['load'].shift(-1, freq = 'H')
# print(train_shifted.head())

for t in range(1, T+1):
    train_shifted[str(T-t)] = train_shifted['load'].shift(T-t, freq = 'H')

X_cols = ['load_t-5', 'load_t-4', 'load_t-3', 'load_t-2', 'load_t-1', 'load_t']
y_cols = 'y_t+1'
train_shifted.columns = ['original']+[y_cols]+X_cols
# print(train_shifted.head())

#결측값 처리
train_shifted = train_shifted.dropna(how='any')

#5단계 넘파이 어레이배열 (샘플,피쳐)로 바꾸기
X_train = train_shifted[X_cols].as_matrix()
y_train = train_shifted[y_cols].as_matrix()

X_train = X_train.reshape(X_train.shape[0], T, 1)

look_back_date = dt.datetime.strptime(validation_start_date, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours = T-1)


# print(look_back_date)
valid = energy.copy()[(energy.index >= look_back_date) & (energy.index < test_start_date)][['load']]
# print(valid.shape)

valid['load'] = scaler.transform(valid)
print(valid.head())

valid_shifted = valid.copy()
valid_shifted['y+1'] = valid_shifted['load'].shift(-1, freq='H')


for t in range(1, T+1):
  valid_shifted['load_t-' + str(T-t)] = valid_shifted['load'].shift(T-t, freq='H')
valid_shifted.head()


valid_shifted = valid_shifted.dropna(how = 'any')
y_valid = valid_shifted['y+1'].as_matrix()
X_valid = valid_shifted[['load_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()

X_valid = X_valid.reshape(X_valid.shape[0], T, 1)

unit_dim = 5
batch_size = 32
epochs = 50

model = Sequential()
model.add(GRU(unit_dim, input_shape = (T,1)))
model.add(Dense(HORIZON))
model.compile(optimizer='RMSprop', loss='mse')
model.summary()

earlystop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(X_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   validation_data=(X_valid, y_valid),
                   callbacks=[earlystop])


##########################################################


look_back_date = dt.datetime.strptime(test_start_date, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
look_back_date
test = energy.copy()[look_back_date:][['load']]
test['load'] = scaler.transform(test)
test_shifted = test.copy()
test_shifted['y+1'] = test_shifted['load'].shift(-1, freq='H')
test_shifted.head()
for t in range(1, T+1):
  test_shifted['load_t-' + str(T-t)] = test_shifted['load'].shift(T-t, freq='H')
test_shifted
test_shifted = test_shifted.dropna(how='any')
y_test = test_shifted['y+1'].as_matrix()
X_test = test_shifted[['load_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()


X_test = X_test.reshape(X_test.shape[0], T, 1)

# %matplotlib inline
predictions = model.predict(X_test)
eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
eval_df['timestamp'] = test_shifted.index
eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
eval_df['actual'] = np.array(np.transpose(y_test)).ravel()
eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
eval_df.head()
mape(eval_df['prediction'], eval_df['actual'])
print(mape(eval_df['prediction'], eval_df['actual']))
eval_df[eval_df.timestamp < '2014-11-08'].plot(x = 'timestamp', y = ['prediction', 'actual'], style = ['r', 'b'], figsize = (15,8))
plt.show()
