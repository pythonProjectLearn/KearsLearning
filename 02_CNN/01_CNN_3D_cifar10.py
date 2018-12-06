# encoding:utf-8
"""卷积网络：用于图片多分类
1、原始数据
X_train.shape  (50000, 32, 32, 3)
单个样本:
array([[[ 26,  17,  13],
        [ 13,  13,  14],
        [ 14,  15,  14],
        ...,
        [ 12,  15,  21],
        [ 36,  26,  22],
        [ 17,  25,  31]],

       [[ 13,  17,  14],
        [ 14,  11,   9],
        [ 19,  18,  11],
        ...,
        [174, 229, 249],
        [251, 244, 248],
        [175,  29,  11]],

    ])

y_train.shape (50000, )
单个： 8


2、标准化,除以255，使得所有的点在（0,1）之间
X_train.shape (50000, 32, 32, 3)
单个样本：
array([[[ 0.10196079,  0.06666667,  0.05098039],
        [ 0.05098039,  0.05098039,  0.05490196],
        [ 0.05490196,  0.05882353,  0.05490196],
        ...,
        [ 0.04705882,  0.05882353,  0.08235294],
        [ 0.14117648,  0.10196079,  0.08627451],
        [ 0.06666667,  0.09803922,  0.12156863]],
       [[ 0.05098039,  0.06666667,  0.05490196],
        [ 0.05490196,  0.04313726,  0.03529412],
        [ 0.07450981,  0.07058824,  0.04313726],
        ...,
        [ 0.68235296,  0.89803922,  0.97647059],
        [ 0.98431373,  0.95686275,  0.97254902],
        [ 0.68627453,  0.11372549,  0.04313726]],
     ])


Y_train.shap  (50000, 10)
单个：
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.])


X_train中每个元素是0-255的数字组成
Y_train中的每个元素是由0或1的数字组成，由10种分类，所以shape是10
多分类标签要one-hot编码

"""
#from __future__ import print_function, division

import keras
from keras import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense,Conv2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
BASEDIR = os.path.dirname(os.path.abspath(__file__))


data_base = '/home/zt/Documents/Data'
cifar10_data = 'cifar-10-batches-py'

def load_data(data_base, pakagepath):
    """加载数据"""
    for abspath, b, filenames in os.walk(os.path.join(data_base,pakagepath)):
        trainpaths = [os.path.join(abspath, filename) for filename in filenames if filename.startswith('data_batch')]
        testpaths = [os.path.join(abspath, filename) for filename in filenames if filename.startswith('test')]

    # train
    with open(trainpaths[0], 'rb') as f:
        d = pickle.load(f, encoding='iso-8859-1')
        data = d['data']
        labels = d['labels']

    for i, filepath in enumerate(trainpaths[1:]):
        with open(filepath, 'rb') as f:
            d = pickle.load(f, encoding='iso-8859-1')
            data0 = d['data']
            labels0 = d['labels']

        data = np.concatenate((data, data0), axis=0)
        labels = np.concatenate((labels, labels0), axis=0)

    data = data.reshape(data.shape[0], 32, 32, 3)

    # test
    with open(testpaths[0], 'rb') as f:
        d = pickle.load(f, encoding='iso-8859-1')
        test_data = d['data']
        test_labels = np.array(d['labels'])
    test_data = test_data.reshape(test_data.shape[0], 32, 32, 3)

    return (data, labels), (test_data, test_labels)


#-------------------------#
def CNN_model(input_shape):
    model = Sequential()

    # 第一层是要指定input_shape，告诉模型输入图片的size
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=input_shape))
    model.add(Activation(activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(Activation(activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 全链接层
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def model_train(X_train, y_train, X_test, y_test, batch_size, epochs, model_path, data_augmentation=True):

    # 不对图片进行裁剪
    if not data_augmentation:
        print('Not using data augmentation.')
        model = CNN_model(X_train.shape[1:])
        history = model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  )
    else: # 对图片进行裁剪预处理
        print('Using real-time data augmentation.')

        datagen = ImageDataGenerator(featurewise_center=False,  # 所有样本的均值为0
                                     samplewise_center=False,   # 每个样本的均值为0
                                     featurewise_std_normalization=False, # 所有样本除以标准差
                                     samplewise_std_normalization=False,  # 每个样本除以标准差
                                     zca_whitening=False, # 应用白化
                                     rotation_range=0, # 随机旋转角度（0,180）
                                     width_shift_range=0.1,  # 随机水平移动，宽度的0.1倍
                                     height_shift_range=0.1,  # 随机上下移动，高度的0.1倍
                                     horizontal_flip=True,  # 随机水平翻转
                                     vertical_flip=False,   # 随机垂直翻转
                                     )
        datagen.fit(X_train)
        model = CNN_model(X_train.shape[1:])
        history = model.fit_generator(generator=datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=5000,  # 每步epochs迭代steps_per_epoch次
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            workers=4,
                            verbose=1
                            )
    return model, history



if __name__=="__main__":
    # 固定参数
    num_classes=10  # 图片有10种分类
    save_dir = BASEDIR +'/Model'
    model_name = 'keras_cifar10_trained_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    (X_train, y_train), (X_test, y_test) = load_data(data_base, cifar10_data)
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255

    print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # 2 训练并获得模型得分
    model, history = model_train(X_train, y_train, X_test, y_test,batch_size=32,epochs=100, model_path=os.path.join(save_dir, model_name))
    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


    # 3 保存模型结构和权重
    model_json = model.to_json()
    open('cifar10_architecture.json', 'w').write(model_json)
    model.save_weights('cifar10_weights.h5', overwrite=True)

    # 4 可视化训练的结果
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
