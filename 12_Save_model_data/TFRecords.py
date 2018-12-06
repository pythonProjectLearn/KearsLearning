# encoding:utf-8
import numpy as np

import tensorflow as tf
import keras
from keras import backend as K
from keras import layers

from tensorflow.contrib.learn.python.learn.datasets import mnist



def cnn_layers(x_train_input):
    x = layers.Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(x_train_input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x_train_out = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return x_train_out


sess = K.get_session()
# 获得批度数据
batch_size = 128
batch_shape = (batch_size, 28, 28, 1)
steps_per_epoch = 469
epochs = 5
num_classes = 10
capacity = 10000
min_after_dequeue = 3000
enqueue_many = True

data = mnist.load_mnist()
x_train_batch, y_train_batch = tf.train.shuffle_batch(
    tensors=[data.train.images, data.train.labels.astype(np.int32)],
    batch_size=batch_size, # 批度
    capacity=capacity,  # 容量变量，控制队列的大小
    min_after_dequeue=min_after_dequeue, # 队列中，元素最小的个数
    #当enqueue_many=True时，使用批度训练，输入的tensor是[*, x,y,z] 输出的将是[batch, x,y,z]
    enqueue_many=enqueue_many,
    num_threads=8  #
)

x_train_batch = tf.cast(x_train_batch, tf.float32)
x_train_batch = tf.reshape(x_train_batch, shape=batch_shape)

y_train_batch = tf.cast(y_train_batch, tf.int32)
y_train_batch = tf.one_hot(y_train_batch, num_classes)

x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

model_input = layers.Input(tensor=x_train_batch)
model_output = cnn_layers(model_input)
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

# Pass the target tensor `y_train_batch` to `compile`
# via the `target_tensors` keyword argument:
train_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[y_train_batch])
train_model.summary()

# Fit the Model using data from the TFRecord data tensors.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

train_model.fit(epochs=epochs,
                steps_per_epoch=steps_per_epoch)

# Save the Model weights.
train_model.save_weights('saved_wt.h5')

# Clean up the TF session.
coord.request_stop()
coord.join(threads)
K.clear_session()

# Second Session to test loading trained Model without tensors
x_test = np.reshape(data.validation.images, (data.validation.images.shape[0], 28, 28, 1))
y_test = data.validation.labels
x_test_inp = layers.Input(shape=(x_test.shape[1:]))
test_out = cnn_layers(x_test_inp)
test_model = keras.models.Model(inputs=x_test_inp, outputs=test_out)

test_model.load_weights('saved_wt.h5')
test_model.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
test_model.summary()

loss, acc = test_model.evaluate(x_test,
                                keras.utils.to_categorical(y_test),
                                batch_size=batch_size)
print('\nTest accuracy: {0}'.format(acc))
