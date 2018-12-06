# encoding:utf-8
from __future__ import print_function

# Sequential组建序列模型，Model用来组合多个model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import UpSampling2D, Conv2D
from keras import layers
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils.generic_utils import Progbar

from collections import defaultdict

import pickle
from PIL import Image

import numpy as np
np.random.seed(1337)
num_classes = 10

def build_generator(latent_size):
    # 生成模型
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    cnn = Sequential()

    # 图片先进入一个全连接层，所以tensor是1维的
    # 后面要跟上卷积层，所以要通过Reshape()将1维的tensor转化成维(7,7,128)的图片维度
    cnn.add(Dense(units=1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(units=128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((7, 7, 128)))

    # upsample to (14, 14, ...)
    # 将数据的行和列分别重复size[0]和size[1]次，所以变成了(14, 14, 128)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(256, 5, padding='same',
                   activation='relu',
                   kernel_initializer='glorot_normal'))

    # upsample to (28, 28, ...)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, 5, padding='same',
                   activation='relu',
                   kernel_initializer='glorot_normal'))

    # take a channel axis reduction
    # 将通道轴降维
    cnn.add(Conv2D(filters=1, kernel_size=2, padding='same',
                   activation='tanh',
                   kernel_initializer='glorot_normal'))


    # this is the z space commonly refered to in GAN papers
    # 定义一个潜空间z，也就是一个随机噪声作为输入数据
    latent = Input(shape=(latent_size, ))

    # this will be our label
    # 定义一个伪造的标签tensor
    image_class = Input(shape=(1,), dtype='int32')
    # Embedding()嵌入层将正整数（下标）转换为具有固定大小的向量
    cls = Flatten()(Embedding(input_dim=num_classes, output_dim=latent_size,
                              embeddings_initializer='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    # z空间latent与分类条件cls用hadamard乘积连接，把latent作为CNN的输入，image_class作为输出，放入cnn模型中进行学习
    h = layers.multiply([latent, cls])

    # 用伪造的随机数据，和伪造的标签，生成伪造的图像fake_image
    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)


def build_discriminator():
    """判别模型"""
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(28, 28, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, padding='same', strides=1))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(28, 28, 1))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    # fake存放判别模型是判别正确了还是判别错了，所以输出维度是1
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    # 保持判别器判别出来的分类是哪个
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

    return Model(image, [fake, aux])

if __name__ == '__main__':

    # batch and latent size taken from the paper
    epochs = 50
    batch_size = 100
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # 1 创建识别模型
    print('Discriminator Model:')
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    discriminator.summary()

    # 2  创建生成模型
    generator = build_generator(latent_size)

    # 3 定义输入和输出的tensor
    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # 4 向生成模型输入潜空间变量latent和图片的类别，生成伪造的图片标签
    fake = generator([latent, image_class])

    # 在联合模型当中，只想训练生成模型，看一下生成模型的损失，不想训练识别模型,
    # 因为在combined模型被训练之前，discriminator已经被训练过了，在后面可以看到
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model([latent, image_class], [fake, aux])
    print('Combined Model:')
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    combined.summary()


    # get our mnist data, and force it to be of shape (..., 28, 28, 1) with
    # range [-1, 1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)

    num_train, num_test = x_train.shape[0], x_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    for epoch in range(1, epochs+1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(x_train.shape[0] / batch_size)
        progress_bar = Progbar(target=num_batches)

        epoch_gen_loss = []  # 生成模型的损失
        epoch_disc_loss = []  # 识别模型的损失

        # 开始训练模型
        for index in range(num_batches):
            # 1 获得批度的真实训练数据
            image_batch = x_train[index*batch_size:(index+1)*batch_size]
            label_batch = y_train[index*batch_size:(index+1)*batch_size]

            # 2 用噪声和随机生成的label数据作为生成模型的训练数据generated_iages
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))
            sampled_labels = np.random.randint(0, num_classes, batch_size)
            generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)

            # 3 把生成的假图像数据和假标签，混合到真数据与真标签中，让识别模型进行对抗学习
            # 把生成的假图像和真图像放到一起
            x = np.concatenate((image_batch, generated_images))
            # 把假标签和真标签放到一起
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # 初始化，识别模型，对伪造模型是识别错误了，还是失败了
            soft_zero, soft_one = 0.25, 0.75
            y = np.array([soft_one] * batch_size + [soft_zero] * batch_size)

            # 4 将数据带入到识别模型中
            epoch_disc_loss.append(discriminator.train_on_batch(x, [y, aux_y]))

            # 5 生成2*batch_size的数据，看一下生成模型的损失
            # 对于生成模型我们想让所有的{fake, not-fake} labels to say not-fake
            # 也就是说尽量欺骗识别模型
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)
            trick = np.ones(2*batch_size)*soft_one
            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels]))

        # 测试
        print('Testing for epoch {}:'.format(epoch))
        # 1 评价识别模型
        # num_test测试集批度大小
        noise = np.random.uniform(-1, 1, (num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, num_test)
        generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=False)
        x = np.concatenate((x_test, generated_images))
        y = np.array([1]*num_test + [0]*num_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)
        # 评价测试集
        discriminator_test_loss = discriminator.evaluate(x, [y, aux_y], verbose=False)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # 2 评价生成模型
        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, 2 * num_test)

        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)


        # 3 将训练集和测试机的生成模型和识别模型的损失记录下来
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        # 4 save weights every epoch
        generator.save_weights(
            '../Model/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            '../Model/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # 5 打印输出
        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))


        # 6 预测
        # generate some digits to display
        num_rows = 10
        noise = np.random.uniform(-1, 1, (num_rows * num_classes, latent_size))
        sampled_labels = np.array([[i] * num_rows for i in range(num_classes)]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict([noise, sampled_labels], verbose=0)

        # prepare real images sorted by class label
        real_labels = y_train[(epoch - 1) * num_rows * num_classes: epoch * num_rows * num_classes]
        indices = np.argsort(real_labels, axis=0)
        real_images = x_train[(epoch - 1) * num_rows * num_classes:   epoch * num_rows * num_classes][indices]

        # 展示生成的图片
        img = np.concatenate(
            (generated_images,
             np.repeat(np.ones_like(x_train[:1]), num_rows, axis=0),
             real_images))

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(img, 2 * num_classes + 1)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            '../Model/plot_epoch_{0:03d}_generated.png'.format(epoch))

    # 7最后保留模型
    pickle.dump({'train': train_history, 'test': test_history},
                open('../Model/acgan-history.pkl', 'wb'))













