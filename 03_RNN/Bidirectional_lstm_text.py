# encoding:utf-8
"""
情感分类:二分类

1、原始数据
x_train.shape (25000,)
单个样本如下，每个数字代表单词在字典中的索引：
[1, 14, 20, 16, 527, 10,.....]

y_train.shape (25000,)
单个形如：0

2、结构化之后
x_train shape: (25000, 400) 把每个样本的单词索引补全成相同的长度
形如：
array([

           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     1,    24,    60,   290,
         149,    14,  5237,  1361,  4929,   285,    44, 53900,     4,
         105,   306,   220, 24814,    34,     4,   862,   398,   769,
           5,    36,   364,   489,   547,     8,     4,   365,    12,
          16,    55,   695,    93,    18,   248,   246,    48,    13,
         258,    12,    23,    61,   699,    13,    62,   509,    12,
         730,   120,     4,   476,    11,     4,    22,    81,    35,
        2659,   292,   246,     4,  1473,   116,     9,  1297,     5,
          38,     4,    20,   996,     8,    66,  2514,    25,    11,
          14,    22,  3674,     4,   228,   837,  1396,  1584,    93,
           4,   204,   530,     5,   371,   127,   193,     4,  4091,
           7,     4,  3825,    75,    32,   119,  2820,    17,  3825,
           9,    33,   211,   616,     5,   120,   917,    25,   566,
        2522,   624,    15,    59,   371,     9,  3825,     4,  3855,
          11,   109,  1267,  3501,     8, 16038,    72,   469,     4,
        5936,     7,     4,    22,   619,     8,   135,    14,   755,
          16,     6,   964,  4060], dtype=int32)

y_train shape: (25000, )
单个形如：0



用双向lstm进行分类:
学习一条句子，从正向和反向两个方向学习句子的特征


双向LSTM看起来比fasttext效果还要好
"""
from __future__ import print_function


from keras.models import Sequential
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.layers import Dense,Embedding, Bidirectional, LSTM, Dropout

import numpy as np

def Bidirec_lstm(max_features, maxlen):
    model = Sequential()
    # Embedding是把一个样本长度为input_length(形如：[1,2,3,4...])，映射成为(input_length, input_dim)的样本
    # 也就是把一个单词的索引映射成为长度为input_dim的向量
    # 随后才能使用LSTM(也就是说输入的样本必须是(None, input_length, input_dim)才能放入LSTM中)
    model.add(Embedding(input_dim=max_features, output_dim=128, input_length=maxlen))
    # 双向
    model.add(Bidirectional(LSTM(units=64)))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__=='__main__':
    max_features = 20000
    maxlen = 100
    batch_size = 32


    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print('x_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    print('Train...')
    model = Bidirec_lstm(max_features=20000, maxlen=100)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=4,
              validation_data=[x_test, y_test])




