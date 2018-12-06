# encoding:utf-8

from __future__ import print_function, division, unicode_literals

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
import psycopg2

def postgresql(sql):
    '''

    sql = """select datetime_stock, close_price from  stock_sh_a where name_stock='上海三毛' """
    '''
    with psycopg2.connect(database='zhoutao', user='postgres', password='postgres', host='127.0.0.1', port='5432') as conn:
        cur = conn.cursor()
        cur.execute(sql)
        data = cur.fetchall()
    return data

def load_data(sql, seq_len, test_ratio, normal_windows=True):
    data = postgresql(sql)
    data_array2 = np.array([data[i:i+seq_len+1] for i in xrange(len(data) - seq_len)])

    if normal_windows:

        X = np.apply_along_axis(lambda x:x/x[0] -1 , 1, data_array2[:, :-1])
        y = data_array2[:, -1]
    else:
        X = data_array2[:, :-1]
        y = data_array2[:,-1]

    split_i = int(len(X)*(1-test_ratio))

    X_test = X[split_i:, :]
    y_test = y[split_i:]

    return X, y, X_test, y_test



def build_model(layers):
    model = Sequential()

    model.add(LSTM(input_shape=(layers[1], layers[0]), output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)


    return prediction_seqs



def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


if __name__=='__main__':
    start_time = time.time()
    print('> Loading data... ')
    seq_len = 50
    sql = """select close_price from  stock_sh_a where name_stock='上海三毛' """
    X_train, y_train, X_test, y_test = load_data(sql, seq_len=seq_len, test_ratio=0.2, normal_windows=True)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print('> Data Loaded. Compiling...')

    model = build_model([1, 50, 100, 1])

    model.fit(X_train,  y_train, batch_size=64, nb_epoch=100)

    predictions = predict_sequences_multiple(model, X_test, seq_len, 50)
    #predicted = lstm.predict_sequence_full(Model, X_test, seq_len)
    #predicted = lstm.predict_point_by_point(Model, X_test)

    print('Training duration (s) : ', time.time() - start_time)
    plot_results_multiple(predictions, y_test, 50)