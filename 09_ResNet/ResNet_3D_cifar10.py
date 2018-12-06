# encoding:utf-8
"""ResNet残差网络
多分类：一共有10个分类，
与多标签不同，多分类是一个样本只对应一个标签，而多标签预测是一个样本有多个标签组合起来确定一个样本的类别

X_train.shape (None, 32, 32, 3)
Y_train.shap  (None, 10)

X_train中每个元素是0-255的数字组成
Y_train中的每个原始是0-9的数字组成

"""
from __future__ import print_function, division

import keras
from keras.layers import Input, Flatten,Dense, Activation, Conv2D, BatchNormalization, AveragePooling2D
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cPickle
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
        d = cPickle.load(f)
        data = d['data']
        labels = d['labels']

    for i, filepath in enumerate(trainpaths[1:]):
        with open(filepath, 'rb') as f:
            d = cPickle.load(f)
            data0 = d['data']
            labels0 = d['labels']

        data = np.concatenate((data, data0), axis=0)
        labels = np.concatenate((labels, labels0), axis=0)

    data = data.reshape(data.shape[0], 32, 32, 3)

    # test
    with open(testpaths[0], 'rb') as f:
        d = cPickle.load(f)
        test_data = d['data']
        test_labels = np.array(d['labels'])
    test_data = test_data.reshape(test_data.shape[0], 32, 32, 3)

    return (data, labels), (test_data, test_labels)


def lr_schedule(epoch):
    """根据不同的epoch调整学习率lr"""
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_block(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', conv_first=True):
    """ 输入层
    strides:卷积操作时，窗口移动的尺寸"""
    # 先卷积后标准化
    if conv_first:
        x = Conv2D(filters=num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_initializer='he_normal',      # 卷积核的初始化
                   kernel_regularizer=l2(1e-4))(inputs) # 在卷积操作上加上正则, inputs是输入tensor

        x = BatchNormalization()(x)  # 对卷积层标准化
        if activation:
            x = Activation(activation=activation)(x)  # 再执行激活函数变换
        return x

    # 先标准化，后卷积
    x = BatchNormalization()(inputs)
    if activation:
        x = Activation('relu')(x)
    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """第一个版本的resnet

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    The number of filters doubles when the feature maps size is halved.
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers 卷积层的深度
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        Model (Model): Keras Model instance
    """
    # depth必须是6n+2层
    if (depth - 2) % 6 !=0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    inputs = Input(shape=input_shape)  # 定义了一个张亮

    num_filters = 16
    num_sub_blocks = int((depth-2)/6)

    # 构建一个resnet层
    x = resnet_block(inputs=inputs)

    # 一共要堆叠3个卷积模块
    for i in range(3):
        for j in range(num_sub_blocks):
            # 首个某块的所有resnet层的窗口移动尺度为1
            strides = 1

            # 另外2个模块的第一个resnet层的窗口移动为2
            is_first_layer_but_not_first_block = j ==0 and i >0
            if is_first_layer_but_not_first_block:
                strides = 2

            y = resnet_block(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_block(inputs=y, num_filters=num_filters, activation=None)
            if is_first_layer_but_not_first_block:
                x = resnet_block(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None)

            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = 2 * num_filters

    # 卷积操作完毕再进行池化
    x = AveragePooling2D(pool_size=8)(x)
    # 输出时必须要输出全连接层
    y = Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    Features maps sizes: 16(input), 64(1st sub_block), 128(2nd), 256(3rd)

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        Model (Model): Keras Model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start Model definition.
    inputs = Input(shape=input_shape)
    num_filters_in = 16

    filter_multiplier = 4
    num_sub_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D on input w/o BN-ReLU
    x = Conv2D(num_filters_in,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)

    # Instantiate convolutional base (stack of blocks).
    for i in range(3):
        if i > 0:
            filter_multiplier = 2
        num_filters_out = num_filters_in * filter_multiplier

        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = resnet_block(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             conv_first=False)
            y = resnet_block(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_block(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if j == 0:
                x = Conv2D(num_filters_out,
                           kernel_size=1,
                           strides=strides,
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate Model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_model(X_train, y_train, X_test, y_test, batch_size, epochs, depth, filepath, version, model_type, data_augmentation=True):
    print(model_type)
    if version == 2:
        model = resnet_v2(input_shape=X_train.shape[1:], depth=depth, num_classes=y_train.shape[1])
    else:
        model = resnet_v1(input_shape=X_train.shape[1:], depth=depth, num_classes=y_train.shape[1])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule) # 随着迭代的深入学习率也跟着改变
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    callbacks = [checkpoint, lr_scheduler, lr_reducer]


    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the Model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=1000,
                            validation_data=(X_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    return model


if __name__=='__main__':
    # 固定参数
    num_classes=10  # 图片有10种分类

    version = 2
    depth = 3 * 6 + 2
    model_type = 'ResNet%dv%d' % (depth, version)
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    (X_train, y_train), (X_test, y_test) = load_data(data_base, cifar10_data)
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)
    X_train = X_train.astype('float32')/ 255
    X_test = X_test.astype('float32')/255
    # 去掉均值
    X_train_mean = np.mean(X_train, axis=0)
    X_train -= X_train_mean
    X_test -= X_train_mean


    print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model = train_model(X_train, y_train, X_test, y_test, batch_size=32, epochs=200,
                        depth=depth, filepath=os.path.join(save_dir, model_name),
                        version=version, model_type=model_type)

    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])





