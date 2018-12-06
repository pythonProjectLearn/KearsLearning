# encoding:utf-8
"""
灰度图

调用sklearn为keras模型调参
当超参树比较少是，可以使用GridSearch()进行调参，精度高，但是耗时长
参数很多时，使用RandGridSearch()进行调参，可以减少参数被重复测试
当然还有贝叶斯调参

用keras封装成sklearn的分类期和回归器，才能使用sklearn中对分类器回归期的操作的函数
"""
import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense,Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

def make_model(dense_layer_sizes, filters, kernel_size, pool_size):
    '''Creates Model comprised of 2 convolutional layers followed by dense layers

    dense_layer_sizes: List of layer sizes.
        This list has one number for each layer
    filters: Number of convolutional filters in each convolutional layer
    kernel_size: Convolutional kernel size
    pool_size: Size of pooling area for max pooling

    被sklearn调用的模型必须要有1、模型结构；2模型编译两块
    '''

    model = Sequential()
    model.add(Conv2D(filters, kernel_size,
                     padding='valid',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

if __name__=='__main__':
    img_rows, img_cols = 28, 28
    num_classes = 10

    # 特征数据预处理，
    # 首先是样把数据的shape规整成相同的shape
    # 其次每个数据的类型必须规整成同一类型
    # 第3需要对数据标准化，要么放缩到（0-1），要么标准化
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # 多分类标签，要one-hot编码
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    input_shape = (img_rows, img_cols, 1)
    # 只调一个超参
    dense_size_candidates = [[32], [64], [32, 32], [64, 64]]
    my_classifier = KerasClassifier(make_model, batch_size=32)
    # param_grid()传给模型的参数都必须是list包裹的参数，不能是单独的一个值，否则它不能识别
    validator = GridSearchCV(my_classifier,
                             param_grid={'dense_layer_sizes': dense_size_candidates,
                                         # epochs is avail for tuning even when not
                                         # an argument to Model building function
                                         'epochs': [3, 6],
                                         'filters': [8],
                                         'kernel_size': [3],
                                         'pool_size': [2]},
                             scoring='neg_log_loss',
                             n_jobs=1)
    validator.fit(x_train, y_train)

    # 获得最好的参数
    print('The parameters of the best Model are: ')
    print(validator.best_params_)
    # 获得最好的模型
    best_model = validator.best_estimator_.model
    metric_names = best_model.metrics_names
    metric_values = best_model.evaluate(x_test, y_test)
    for metric, value in zip(metric_names, metric_values):
        print(metric, ': ', value)