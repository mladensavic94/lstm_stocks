import tkinter
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

def readData():
    tkinter.Tk().withdraw()
    filename = askopenfilename(initialdir="individual_data", filetypes=[("CSV file", "*.csv")])
    file = pd.read_csv(filename)
    return file

def preprocessData(inputData, scaler):
    data = np.array(inputData)
    data  = data[:,1:5]
    data = np.array(data).reshape((data.shape[0], data.shape[1]))
    data = scaler.fit_transform(data)
    return data

def splitData(data, percentage=0.8):
    return train_test_split(data, test_size= 1 - percentage, shuffle = False)

def plotTestAndTrain(train, test):
    train_len = len(train)
    test_len = len(test)
    plt.subplot(221)
    plt.title("Open")
    plt.plot(range(0, train_len), train[:,0], color = 'red')
    plt.plot(range(train_len, test_len + train_len), test[:,0], color = 'blue')

    plt.subplot(222)
    plt.title("High")
    plt.plot(range(0, train_len), train[:,1], color = 'red')
    plt.plot(range(train_len, test_len + train_len), test[:,1], color = 'blue')

    plt.subplot(223)
    plt.title("Low")
    plt.plot(range(0, train_len), train[:,2], color = 'red')
    plt.plot(range(train_len, test_len + train_len), test[:,2], color = 'blue')

    plt.subplot(224)
    plt.title("Close")
    plt.plot(range(0, train_len), train[:,3], color = 'red')
    plt.plot(range(train_len, test_len + train_len), test[:,3], color = 'blue')
    plt.legend(["train", "test"], loc=7)
    plt.show()

def plotPredictedData(real, predicted):
    plt.subplot(221)
    plt.title("Open")
    plt.plot(real[:,0], color = 'blue')
    plt.plot(predicted[:,0], color = 'red')

    plt.subplot(222)
    plt.title("High")
    plt.plot(real[:,1], color = 'blue')
    plt.plot(predicted[:,1], color = 'red')

    plt.subplot(223)
    plt.title("Low")
    plt.plot(real[:,2], color = 'blue')
    plt.plot(predicted[:,2], color = 'red')

    plt.subplot(224)
    plt.title("Close")
    plt.plot(real[:,3], color = 'blue')
    plt.plot(predicted[:,3], color = 'red')
    plt.legend(["real, predicted"])
    plt.show()    

def prepareForTraining(train, test):
    Y_train = np.roll(train, -1, axis=0)
    train = train[:-1,:]
    train = np.reshape(train, (train.shape[0], train.shape[1],1))
    Y_train = Y_train[:-1,:]
    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1],1))
    y_test = np.roll(test, -1, axis=0)
    test = test[:-1,:]
    test = np.reshape(test, (test.shape[0], test.shape[1],1))
    y_test = y_test[:-1,:]
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1],1))
    return train, Y_train, test, y_test

def train(x_train, y_train, model, batch_size = 8, epochs = 10, file_name = "model.h5"):
    model.fit (x_train, y_train, batch_size = batch_size, epochs = epochs, shuffle = False)
    model.save("model.h5")

def test(model,x_train, y_train,x_test, y_test):
    print(model.summmary())
    score_train = model.evaluate(x_train, y_train, batch_size =1)    
    score_test = model.evaluate(x_test, y_test, batch_size =1)
    print ("train RMSE = ", score_train**0.5) 
    print ("test RMSE = ", score_test**0.5)

def predict(model, test_data, real_data, scalar):
    pred = model.predict(test_data)
    pred = scalar.inverse_transform(np.reshape(pred, (test_data.shape[0],test_data.shape[1])))
    real_data = scaler.inverse_transform(np.reshape(real_data, (real_data.shape[0],real_data.shape[1])))
    plotPredictedData(real_data, pred)


if __name__ == '__main__':
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    train, test = splitData(preprocessData(readData(), scaler))
    x, y, w, q = prepareForTraining(train, test)
    
