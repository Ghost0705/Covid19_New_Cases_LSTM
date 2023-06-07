"""
Covid-19 Cases In Malaysia Dataset
"""
# %%
#import the library
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.utils import plot_model

#%%
PATH = os.getcwd()
CSV_PATH_TRAIN = os.path.join(PATH,"cases_malaysia_train.csv")
CSV_PATH_TEST = os.path.join(PATH,"cases_malaysia_test.csv")
df_train = pd.read_csv(CSV_PATH_TRAIN)
df_test = pd.read_csv(CSV_PATH_TEST)

df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')

# %%
#Data inspection
df_train.head()

#%%
df_train.describe()

#%%
df_train.info()

#%%
df_test.info()
# %%
df_train.isnull().sum()
# %%
df_test.isnull().sum()
#%%
df_train.duplicated().sum()
#%%
df_test.duplicated().sum()

#%%
df_train['cases_new'] = df_train['cases_new'].fillna(df_train['cases_new'].mean())
df_test['cases_new'] = df_test['cases_new'].fillna(df_test['cases_new'].mean())

#%%
df_train.info()

#%%
df_test.info()

# %%
#Feature selection
df_train_new_case = df_train['cases_new']
df_test_new_case = df_test['cases_new']
mms = MinMaxScaler()
df_train_scale = mms.fit_transform(np.expand_dims(df_train_new_case,axis=-1))
df_test_scale = mms.fit_transform(np.expand_dims(df_test_new_case,axis=-1))

# %%
#Data windowing
window_size = 30
X_train = []
y_train = []

for i in range(window_size,len(df_train_new_case)):
    X_train.append(df_train_scale[i-window_size:i])
    y_train.append(df_train_scale[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# %%
#Concatenate train and test data together
data_test = np.concatenate((df_train_scale, df_test_scale))
length_days = window_size + len(df_test_scale)
data_test = data_test[-length_days:]

X_test = []
y_test = []

for i in range(window_size, len(data_test)):
    X_test.append(data_test[i-window_size:i])
    y_test.append(data_test[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

# %%
#Model development
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape = (X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()
plot_model(model,show_shapes=True,show_layer_names=True)

# %%
#Compile the model
model.compile(optimizer='adam',loss='mse',metrics=['mae'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

# %%
#Model training
history = model.fit(X_train,y_train,epochs=800, validation_data=(X_test,y_test),callbacks= tensorboard_callback)

# %%
#Model evaluation
print(history.history.keys())

# %%
#Deploy the model
y_pred = model.predict(X_test)

# %%
y_true = y_test
y_pred = y_pred.reshape(len(y_pred),1)

# %%
#Plot actual vs predicted
plt.figure()
plt.plot(y_true,color='red')
plt.plot(y_pred,color='blue')
plt.legend(['Actual','Predicted'])

print('\n Mean absolute percentage error: ', mean_absolute_error(y_true, y_pred)/sum(abs(y_true))*100)
# %%
save_path = os.path.join("save_model", "Covid_19_new_case_model.h5")
model.save(save_path)

# %%
