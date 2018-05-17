import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from scipy.ndimage.interpolation import shift
from sklearn import preprocessing
from keras.models import load_model

fname="C:\\Workspaces\\python\\individual_data\\GOOGL_data.csv"
data_csv = pd.read_csv(fname)
 
total_data = len(data_csv)
train_data_len = round(total_data*0.9)

data = np.array(data_csv)
data  = data[:,1:5]
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
data = np.array(data).reshape((data.shape[0], data.shape[1]))
data = scaler.fit_transform(data)
X_train = data[0:train_data_len,:]
x_test = data[train_data_len+1:,:] 
Y_train = np.roll(X_train, -1, axis=0)
X_train = X_train[:-1,:]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
Y_train = Y_train[:-1,:]
Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1],1))
y_test = np.roll(x_test, -1, axis=0)
x_test = x_test[:-1,:]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
y_test = y_test[:-1,:]
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1],1))
model_path = "model.h5"

model = load_model(model_path)
x = scaler.transform(np.array([[65.23, 65.83, 65.02, 65.34]]))
x = x.reshape(1,4,1)
pred1 = model.predict(x) 
print(x)
print(pred1)
pred1 = scaler.inverse_transform(np.reshape(pred1, (1,4)))

print(pred1)
print(scaler.inverse_transform(np.reshape(x, (1,4))))

plt.subplot(221)
plt.title("Open")
plt.plot(scaler.inverse_transform(np.reshape(y_test, (y_test.shape[0],y_test.shape[1])))[:,0], color = 'blue')
plt.plot(pred1[:,0], color = 'red')

plt.subplot(222)
plt.title("High")
plt.plot(scaler.inverse_transform(np.reshape(y_test, (y_test.shape[0],y_test.shape[1])))[:,1], color = 'blue')
plt.plot(pred1[:,1], color = 'red')

plt.subplot(223)
plt.title("Low")
plt.plot(scaler.inverse_transform(np.reshape(y_test, (y_test.shape[0],y_test.shape[1])))[:,2], color = 'blue')
plt.plot(pred1[:,2], color = 'red')

plt.subplot(224)
plt.title("Close")
plt.plot(scaler.inverse_transform(np.reshape(y_test, (y_test.shape[0],y_test.shape[1])))[:,3], color = 'blue')
plt.plot(pred1[:,3], color = 'red')

plt.show()