import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from keras.losses import MeanSquaredError

def minmaxscaler(X, min, max):
    omax, omin = X.max(axis=0), X.min(axis=0)
    
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    
    return X_scaled, omax, omin

def inverse_scalar(X, omax, omin, min, max):
    X = X - min
    X = X / (max - min)
    
    p1 = X + omin
    p2 = omax - omin 
    X = X * (omax - omin)
    X += omin

    return X

def getColumnsData(df, cols):
    print("Retrieving", ' '.join(cols), "Column(s)")
    return df[cols]

def getRequiredColumns(df):
    res = []
    dateColName = None
    closeColName = None

    for col in df.columns:
        if (('date' in col.lower()) or ('time' in col.lower())):
            dateColName = col
            break

    for col in df.columns:
        if ('open' in col.lower()):
            res.append(col)
            break
    
    for col in df.columns:
        if ('low' in col.lower()):
            res.append(col)
            break

    for col in df.columns:
        if ('high' in col.lower()):
            res.append(col)
            break
    
    for col in df.columns:
        if (('close' in col.lower()) and ('adj' not in col.lower()) and ('prev' not in col.lower())):
            res.append(col)
            closeColName = col
            break
    
    for col in df.columns:
        if (('volume' in col.lower()) or ('turnover' in col.lower())):
            res.append(col)
            break

    return res, dateColName, closeColName
    

def LMS(df, pred_col, next_days, epochs, updateEpochs):
    print("LMS Training for", pred_col)
    
    ndf, omax, omin = minmaxscaler(df[pred_col], 1000, 2000)
    x = ndf.values
    
    tmp = []
    for i in x: tmp.append(i)

    x = np.array(tmp)

    def lmsPred(x,l,u,N):
        xd = np.block([1, x]).T
        y=np.zeros((len(xd),1))

        xn = np.zeros((N+1,1))
        xn = np.matrix(xn)

        wn=np.random.rand(N+1,1)/10 
        
        M=len(xd)
        for epoch in range(epochs):
            updateEpochs(epoch)
            print("epoch ", epoch+1, "/", epochs, sep='')

            for n in range(0,M):
                xn = np.block([[xd[n]], [xn[0:N]]])
                y[n]= np.matmul(wn.T, xn)

                if(n>M-l-1): e = 0;
                else: e=int(x[n]-y[n])

                wn = wn + 2*u*e*xn
                
        return y,wn;

    x_train = x[:-next_days]
    u = 2**(-30);

    l=next_days;
    N=100;

    y,wn = lmsPred(x_train,l,u,N)
    
    x = inverse_scalar(ndf, omax, omin, 1000, 2000)
    y = inverse_scalar(y, omax, omin, 1000, 2000)

    json = {
        "inputs": x,
        "outputs": y,
        "actual": x[-l:].values,
        "predicted": y[-l:]
    }

    return json


class EpochPrintingCallback(tf.keras.callbacks.Callback):
    def __init__(self, updateEpochs):
        self.updateEpochs = updateEpochs

    def on_epoch_end(self, epoch, logs=None):
        print(epoch)
        self.updateEpochs(epoch)

import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def LSTMAlgorithm(df, fileName, train_size, epochs, updateEpochs):
    # Preprocess to handle commas in numeric strings
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if column contains strings
            df[column] = df[column].str.replace(',', '').astype(float, errors='ignore')

    cols, dateColName, trade_close_col = getRequiredColumns(df)

    scaling_data_frame = df.filter(cols)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_Data = scaler.fit_transform(scaling_data_frame)
    scaled_data_frame = pd.DataFrame(data=scaled_Data, index=df.index, columns=cols)

    stock_close_data = df.filter([trade_close_col])
    stock_close_dataset = stock_close_data.values

    trainingDataLength = math.ceil(len(stock_close_dataset) * train_size)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(stock_close_dataset)

    StockTrainData = scaledData[0:trainingDataLength, :]

    Xtrain = []
    Ytrain = []

    for i in range(60, len(StockTrainData)):
        Xtrain.append(StockTrainData[i-60:i, 0])
        Ytrain.append(StockTrainData[i, 0])

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))

    print("\n\nLSTM Algorithm for " + str(epochs) + " epochs")

    neurons = 50
    
    model = Sequential()

    model.add(LSTM(neurons, return_sequences=True, input_shape=(Xtrain.shape[1], 1)))
    model.add(LSTM(neurons, return_sequences=False)) 

    model.add(Dense(25)) 
    model.add(Dense(1))

    model.compile(optimizer='adam', loss=MeanSquaredError()) 

    history_data = model.fit(Xtrain, Ytrain, 
                            batch_size=50, epochs=epochs, validation_split=0.2, 
                            verbose=0, callbacks=[EpochPrintingCallback(updateEpochs=updateEpochs)])
    print("Saving Model--------------------------------------------->")
    
    local_pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')
    os.makedirs(local_pretrained_path, exist_ok=True)
    model.save(os.path.join(local_pretrained_path, f'{fileName}.h5'))

    return model


def getPredictonsFromModel(fileName, train_size):
    local_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    df = pd.read_csv(os.path.join(local_dataset_path, f'{fileName}.csv'))
    
    # Preprocess to handle commas in numeric strings
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace(',', '').astype(float, errors='ignore')

    cols, dateColName, trade_close_col = getRequiredColumns(df)

    if not trade_close_col or len(cols) < 5:  # Ensure required columns exist
        return {"error": "Invalid dataset: Missing required columns (date, open, high, low, close, volume)"}

    local_pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')
    model = tf.keras.models.load_model(os.path.join(local_pretrained_path, f'{fileName}.h5'))

    scaling_data_frame = df.filter(cols)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_Data = scaler.fit_transform(scaling_data_frame)
    scaled_data_frame = pd.DataFrame(data=scaled_Data, index=df.index, columns=cols)

    stock_close_data = df.filter([trade_close_col])
    stock_close_dataset = stock_close_data.values

    trainingDataLength = math.ceil(len(stock_close_dataset) * train_size)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(stock_close_dataset)

    StockTrainData = scaledData[0:trainingDataLength, :]

    Xtrain = []
    Ytrain = []

    for i in range(60, len(StockTrainData)):
        Xtrain.append(StockTrainData[i-60:i, 0])
        Ytrain.append(StockTrainData[i, 0])

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    testingData = scaledData[trainingDataLength - 60:, :]

    Xtest = []
    Ytest = stock_close_dataset[trainingDataLength:, :]
    for i in range(60, len(testingData)):
        Xtest.append(testingData[i-60:i, 0])

    Xtest = np.array(Xtest)
    if len(Xtest) == 0:
        return {"error": "Insufficient data for prediction: Dataset too small or training split too large"}

    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

    predictions = model.predict(Xtest)
    predictions = scaler.inverse_transform(predictions).flatten()  # Flatten the 2D array to 1D

    training = stock_close_data[:trainingDataLength]
    validation = pd.DataFrame(df[trade_close_col][trainingDataLength:], columns=['Close'], index=df.index[trainingDataLength:])

    if len(predictions) != len(validation):
        return {"error": f"Prediction length mismatch: Expected {len(validation)} predictions, got {len(predictions)}"}

    validation['Predictions'] = predictions

    real = validation['Close'].values
    pred = validation['Predictions'].values
    n = len(pred)

    accuracy = 0
    for i in range(n):
        accuracy += (abs(real[i] - pred[i])/real[i])*100

    accuracyPercentage = 100 - accuracy/n

    trainingDates = df[dateColName].iloc[:trainingDataLength]
    trainingDates = list(trainingDates.values)
    trainingData = list(training[trade_close_col].values)
    
    realData = list(real)
    
    predictionDates = df[dateColName].iloc[trainingDataLength:]
    predictionDates = list(predictionDates.values)
    predictionData = list(pred)

    for i in range(len(trainingData)): trainingData[i] = float(trainingData[i])
    for i in range(len(predictionData)): predictionData[i] = float(predictionData[i])

    json = {
        "training": {
            "dates": trainingDates,
            "data": trainingData
        },
        "predictions": {
            "dates": predictionDates,
            "realData": realData,
            "predictedData": predictionData,
            "accuracy": accuracyPercentage
        }
    }

    return json


def getManualPredictionForModel(fileName, train_size, openValue, highValue, lowValue, volumeValue):
    local_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    df = pd.read_csv(os.path.join(local_dataset_path, f'{fileName}.csv'))
    
    # Preprocess to handle commas in numeric strings
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace(',', '').astype(float, errors='ignore')

    cols, dateColName, trade_close_col = getRequiredColumns(df)

    close_idx = -1
    for col in df.columns:
        close_idx += 1
        if(col == trade_close_col): break
    
    row = []
    for i in range(df.shape[1]):
        if(i==close_idx): row.append(random.randint(int(float(lowValue)), int(float(highValue))))
        else: row.append(0)
    df.loc[df.shape[0]] = row

    local_pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')
    model = tf.keras.models.load_model(os.path.join(local_pretrained_path, f'{fileName}.h5'))

    scaling_data_frame = df.filter(cols)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_Data = scaler.fit_transform(scaling_data_frame)
    scaled_data_frame = pd.DataFrame(data=scaled_Data, index=df.index, columns=cols)

    stock_close_data = df.filter([trade_close_col])
    stock_close_dataset = stock_close_data.values

    trainingDataLength = math.ceil(len(stock_close_dataset) * train_size) - 1

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(stock_close_dataset)

    testingData = scaledData[trainingDataLength - 60:, :]

    Xtest = []
    for i in range(60, len(testingData) + 1):
        Xtest.append(testingData[i-60:i, 0])

    Xtest = np.array(Xtest)
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

    predictions = model.predict(Xtest)
    predictions = scaler.inverse_transform(predictions)

    return predictions[-1][0]