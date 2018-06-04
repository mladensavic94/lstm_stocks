import common
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

def getModel():
    model = Sequential ()
    model.add(LSTM(1000 , activation = 'tanh', input_shape=(4,1),return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(500 , activation = 'tanh', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(200 , activation = 'tanh',return_sequences=True))
    model.add(Dense (1, activation ='linear'))
    rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.8, epsilon=1e-08)
    model.compile (loss ="mean_squared_error" , optimizer = rmsprop)
    return model 

if __name__ == '__main__':
    file = common.readData()
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data = common.preprocessData(file, scaler)
    train_data, test_data = common.splitData(data, 0.8)
    common.plotTestAndTrain(train_data, test_data)
    x_train, y_train, x_test, y_test = common.prepareForTraining(train_data, test_data)
    model = getModel()
    common.train(x_train, y_train, model, 8, 10, "model02.h5")
    common.test(model, x_train, y_train, x_test, y_test)
    common.predict(model,x_test, y_test, scaler)

    