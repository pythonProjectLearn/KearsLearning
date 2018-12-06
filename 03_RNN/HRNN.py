# encoding:utf-8
"""
Hierarchical RNN (HRNN):
HRNN能在一个复杂的序列上，跨越多个级别的时态层次结构。
一般，HRNN的第一层是编码一个句子（即单词的词向量）成为sentence vector，第二层编码这个sentence vector成为一个document vector
这个文档向量被认为是可以保留词向量和句子结构的

在mnis中第1层LSTM将每每列为shape(28, 1)的pixels编码成列向量shape(128,0)
第2层lstm将shape为（28， 128）的28列，编码成image vetor表示整个image


# TimeDistributed包装器
把一个层应用到输入的每一个时间步上。输入至少为3D张量，下标为1的维度将被认为是时间维
# 1 包装LSTM：
    encoded_rows = TimeDistributed(LSTM(units=row_hidden))(x)
# 2 包装Dense：以产生针对各个时间步信号的独立全连接：
    Model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))  # 输出(None, 10, 8)
    




"""
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM

num_classes = 10


# The data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshapes data to 4D for Hierarchical RNN.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Converts class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

row, col, pixel = x_train.shape[1:]


def HRNN(row_hidden = 128,col_hidden = 128):
    # 输入4D
    x = Input(shape=(row, col, pixel))
    # TimeDistributed把一个层应用到输入的每一个时间步上
    # 使用TimeDistributed编码pixel的每一行,这里把pixel当做时间维度
    encoded_rows = TimeDistributed(LSTM(units=row_hidden))(x)
    encoded_columns = LSTM(col_hidden)(encoded_rows)
    prediction = Dense(num_classes, activation='softmax')(encoded_columns)
    model = Model(x, prediction)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

if __name__=='__main__':
    batch_size = 32
    epochs = 5
    model = HRNN()
    # Training.
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # Evaluation.
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
