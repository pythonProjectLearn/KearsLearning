# encoding:utf-8
"""
fasttext的模型简单，训练速度快，效果最好；
CNN+LSTM模型复杂效果次之，
LSTM效果最差

1、原始数据
x_train.shape (25000,)
单个样本如下，每个数字代表单词在字典中的索引：
[1, 14, 20, 16, 527, 10,.....]

y_train.shape (25000,)
单个形如：0

2、 n-gram之后
X_train.shape (25000, 400)
单个样本：
array([
            14,      20,      16,     527,      10,      10,      36,
           161,    2061,     101,       7,       4,     105,      33,
            32,       5,       4,     769,      16,     256,      46,
          2357,      12,      16,       6,    3777,    8838,    1390,
           535,       4,     206,     139,      23,       6,      20,
            40,      14,       8,      30,      94,     565,     757,
            21,    1095,    1650,    2459,      25,      19,      60,
             6,     387,     180,      11,      15,    2550,      10,
            10,      82,       4,     116,      16,      43,       6,
          1491,     752,       4,     651,       7,       6,     364,
           352,    4522,     509,       5,      13,     615,     384,
            15,      10,      10,      13,      16,     165,     654,
             8,      67,       4,     130,     898,      23,      14,
            31,    1203,      12,      16,      43,      15,      78,
           591,     845,      25,      81,      84,      92,     437,
           129,      58,       5,     278,      23,       6,    2136,
            20,      40,    1095,    1650,  920252,  140646,  143912,
       1020083,  836662,  141097,  562390, 1038284,  819176, 1189078,
        571265, 1149090,   55046,  879000,  246662,  460389,  690143,
        385562, 1056287, 1120099,  303071,  314553,  805234,  176781,
        220507,  864666,  459037,  345909,  410597, 1106943,  103939,
        833200,  174766, 1077768,  173504, 1160578,  802429, 1095147,
        713955,  759596, 1007029,  482731,  907502,  657339,  268088,
        970556,  255528, 1122334,  166297,  118743, 1062145,  809618,
        109552,  197012, 1194760,  610114,  141097,  821868,   51859,
         90948,  674531, 1174380,  815750,   30542,  595738,  466763,
        492426,  534526, 1187952, 1025654, 1174207,  169573,   62094,
        100015,  231845,  524811,  695529, 1033335,  769238,  141097,
        713734, 1156005,  265049,  530065,  848711,  175022, 1135365,
        321622, 1079035,  604391,  923545,  866273,  757355,   35201,
        176781, 1174380, 1164940,  735443,  507712,  144775,  422309,
       1117974,  303017, 1011455,   81887,   79881,  224006,  233260,
        515201,  625523, 1077768,  903141,  320920, 1160578,  534814,
        268088], dtype=int32)


y_train.shape (25000,)
单个形如：0


"""
from __future__ import print_function,division

from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, GlobalAveragePooling1D

import numpy as np

# 创建文本特征,N-gram的特征
def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences



def train_model(x_train, y_train, x_test, y_test,
                max_features=20000,maxlen=400,embedding_dims=50,
                batch_size=32,epochs=5):
    model = Sequential()
    model.add(Embedding(max_features,  embedding_dims, input_length=maxlen))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(GlobalAveragePooling1D())

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))


if __name__=="__main__":
    ngram_range = 2
    max_features = 20000
    maxlen = 400
    batch_size = 32
    embedding_dims = 50
    epochs = 5

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # 建立词库ngram_set中，其中存放的是每个ngram特征
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # 给每个ngram特征添加索引
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # 计算词特征的总数,原来的max_features参数被改变了
        # 把0 作为在训练集中没有出现的单词
        max_features = np.max(list(indice_token.keys())) + 1

        # 通过索引将文本构建ngram特征
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)

    train_model(x_train, y_train, x_test, y_test,max_features=max_features)