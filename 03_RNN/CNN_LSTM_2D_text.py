# encoding:utf-8
"""
(25000, 'train sequences')
(25000, 'test sequences')
Pad sequences (samples x time)
('x_train shape:', (25000, 100))
('x_test shape:', (25000, 100))

"""
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.datasets import imdb


# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build Model...')

def train_model(max_features=20000, maxlen=100,embedding_size=128,
                filters=64, kernel_size=5,
                pool_size=4,
                lstm_output_size=70,
                batch_size=30, epochs=2):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_size, input_length=maxlen))

    # 词向量化之后，通过卷积池化操作提取词的特征
    # 因为词向量化之后，shap是2D的，所以卷积要用1D
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))

    # 卷积池化提取特征之后，进入LSTM层
    model.add(LSTM(units=lstm_output_size))

    # LSTM之后总是接一个全链接层
    model.add(Dense(units=1))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__=='__main__':
    train_model()
