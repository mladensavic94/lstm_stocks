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

plt.subplot(221)
plt.title("Open")
plt.plot(data[:,0], color = 'red')
plt.plot(X_train[:,0], color = 'blue')

plt.subplot(222)
plt.title("High")
plt.plot(data[:,0], color = 'red')
plt.plot(X_train[:,1], color = 'blue')

plt.subplot(223)
plt.title("Low")
plt.plot(data[:,0], color = 'red')
plt.plot(X_train[:,2], color = 'blue')

plt.subplot(224)
plt.title("Close")
plt.plot(data[:,0], color = 'red')
plt.plot(X_train[:,3], color = 'blue')

plt.show()

#shit y to left and drop last
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



 
seed = 666
np.random.seed (seed)
model = Sequential ()
model.add(LSTM(500 , activation = 'tanh', input_shape=(4,1),return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(200 , activation = 'tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100 , activation = 'tanh',return_sequences=True))
model.add(Dense (1, activation ='linear'))


rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.8, epsilon=1e-08)
adam = optimizers.Adam(lr=0.0001)
model.compile (loss ="mean_squared_error" , optimizer = adam)   
model.fit (X_train, Y_train, batch_size = 8, epochs = 30, shuffle = False)
model.save("best01.h5")

print (model.summary())
 
score_train = model.evaluate(X_train, Y_train, batch_size =1)    
score_test = model.evaluate(x_test, y_test, batch_size =1)
print ("train MSE = ", score_train) 
print ("test MSE = ", score_test )
 
    
pred1 = model.predict(x_test) 

pred1 = scaler.inverse_transform(np.reshape(pred1, (x_test.shape[0],x_test.shape[1])))

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
