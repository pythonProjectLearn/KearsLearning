# encoding:utf-8
"""
多标签预测：答案是4位数，每个位数都要预测，所以是多标签

Input: "535+678"  X:one-hot最长编码一个样本的shape是(None, 6, 12)
Output: "1213"    y:one-hot最长编码一个样本的shape是(None, 4, 12)
用model.predict_classes()对one-hot的每个位置当做0-1二分类来预测

Input Output中的每个元素是有0或1组成

535+678=1213
"""
from __future__ import print_function, division, unicode_literals

import numpy as np

class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.chars_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """对字符进行one-hot编码"""
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.chars_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

chars = '0123456789+ '
ctable = CharacterTable(chars)


TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True
MAXLEN = DIGITS + 1 + DIGITS  # 输入的字符长度

questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                    for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    # Skip any addition questions we've already seen
    # Also skip any such that x+Y == Y+x (hence the sorting).
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN.
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1.
    ans += ' ' * (DIGITS + 1 - len(ans))
    if INVERT:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
        # space used for padding.)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

# 大乱数据
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

#
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)
print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

#-----以上全是构造数据，
import keras
from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed

def LSTM_model(HIDDEN_SIZE=128, LAYERS=1):
    model = Sequential()
    model.add(LSTM(units=HIDDEN_SIZE,input_shape=(MAXLEN, len(chars))))
    # RepeatVector层将输入重复n次，也就是说RepeatVector紧跟在输入层后面，
    # 它使得输入层重复n次
    model.add(RepeatVector(DIGITS+1))
    for _ in range(LAYERS):
        # return_sequences=True返回整个序列，False返回整个序列的最后一个
        model.add(LSTM(units=HIDDEN_SIZE, return_sequences=True))
    # 包装器TimeDistributed是为了包装Dense进行全连接的，以产生针对各个时间步信号的独立全连接
    # 因为输入的长度是len(chars)例如535+678"是7位，那么这个样本是(7,12)
    # 因为是按照顺序输入7个字符，要求每个字符按照顺序独立全连接
    model.add(TimeDistributed(Dense(len(chars))))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# 训练模型
model = LSTM_model()
for iteration in range(1, 200):
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_val, y_val))
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        # Model.predict_classes把one-hot的每个位置当做0-1的二分类来预测
        preds = model.predict_classes(rowx, verbose=0)
        # 问题
        q = ctable.decode(rowx[0])
        # 答案
        correct = ctable.decode(rowy[0])

        guess = ctable.decode(preds[0], calc_argmax=False)

        print('Q', q[::-1] if INVERT else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '对' + colors.close, end=' ')
        else:
            print(colors.fail + '错' + colors.close, end=' ')
        print(guess)

