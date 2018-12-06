# encoding:utf-8
"""
序列到序列
本案例是英文翻译成法文

1、原始数据

input_texts 是list,长度为10000
形如：['Go.', 'Run!',.....]

target_texts 是list,长度为10000
形如： ['\tVa !\n', '\tCours\xe2\x80\xaf!\n']
翻译之后，每个字符串以\t开头\n结尾

2、结构化数据之后
encoder_input_data.shape  (10000, 16, 73)
单个样本形如，是one-hot编码：
array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ...,
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]], dtype=float32)



decoder_input_data.shape (10000, 63, 92)
同样是one-hot编码
array([[ 1.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ...,
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]], dtype=float32)


decoder_target_data.shape (10000, 63, 92)
decoder_target_data同decoder_input_data都是法文，但是decoder_input_data的首字符不在decoder_target_data中


3、总结

('Number of samples:', 10000) 一共有1000个样本
('Number of unique input tokens:', 73)  输入的单个词one-hot编码长度是73（英文）
('Number of unique output tokens:', 92) 输出的单个词one-hot编码长度是92（法文）
('Max sequence length for inputs:', 16) 输入的英文文本最大长度是16个单词
('Max sequence length for outputs:', 63)输出的法文最大长度是63个单词

做seq2seq模型翻译时，是将英文与法文分别给予一个编码模型,分别得到自己的编码状态，然后用各自的编码状态去，然后用中间状态作为转化。
就行世界各个语言都以unicode作为中间编码，然后转码成各国的的文字

1、 一个encoder的LSTM层将input sequence编码成2个state vectors(只保留最后一个LSTM的状态， 并丢弃输出)
2、用一个decoder LSTM层被训练用来将目标序列（法文）转化为相同长度的序列，但是在未来要偏离一步。这个训练过程在文本内容上叫“向前学习”
用来初始化来自encoder的state vector是非常有效的.当给定输入一个序列和一个目标序列targets[...t]时，
decoder学习产生'targets[t+1, ...]'

在推断模型中，当我们想decoder不知道的输入序列时，
1、我们要encoder输入序列，得到到一个state vector
2、以一个size=1的target sequence作为开始（即以一个字符1-char作为开始）
3、将state vector 和1-char喂给decoder去预测下一个字符next character
4、从next character中用np.argmax选择概率最大的那个字符作为预测的字符
5、将得到的预测的字符添加到target sequence中
6、重复，直到最后一个字符，或者到了我们规定的字符传的最大长度

# Data download
English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078

"""

import numpy as np


data_base = '/home/zt/Documents/Data'

# -----------------数据预处理成one-hot编码
num_samples = 10000
data_path = data_base+'/fra-eng/fra.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = open(data_path).read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# 1 建立单词所
# 所有英文单词所以
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
# 所有法文单词索引
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

# 2 one-hot编码输入的文本和输出的文本
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# 以上是对文本构造one-hot编码的文本特征

# -------------------------
from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM
from keras.models import Model

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.


# 一、训练阶段

# 1 获得英文的编码输出之后，得到编码输出、中间状态
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# 丢弃encoder_outputs,只保留state（状态），所以要输出状态return_state=True
encoder = LSTM(units=latent_dim, return_state=True)
encoder_outputs,state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# 2 获得法文模型的编码输出，和中间状态
# 由于是在训练阶段，这里要用英文的状态，替换掉法文的状态，来获得法文的训练输出
# 2.1输入层
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# 2.2 lstm层，接收了encoder的状态
# 即要返回内部状态，也要返回完整的序列所以return_sequences=True
decoder = LSTM(latent_dim, return_state=True, return_sequences=True)
# 在训练模型的时候不用返回状态（这里为空），但是在推理的时候用它
# 这里接收了encoder输出的状态
decoder_outputs, _,_ = decoder(decoder_inputs, initial_state=encoder_states)
# 2.3建立全连接层
# 总共有num_decoder_tokens个单词，每个单词当做一个类别，所以用softmax
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 3 训练模型
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x=[encoder_input_data, decoder_input_data], y=decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('s2s.h5')


#------二、推断阶段

reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
# 1 建立编码模型
# 一条新的文本进来，它要经过下面的模型转化才能进行翻译
# 输入一条样本，输出编码的状态
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

# 2 建立解码模型
# 输入解码的样本和状态，输出解码的样本和解码状态
# 2.1 首先要得到解码的状态
decoder_state_input_h = Input(shape=(latent_dim, ))
decoder_state_input_c = Input(shape=(latent_dim, ))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# 训练的时候没有要状态，推断的时候保留状态
decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
# 全连接层
decoder_outputs = decoder_dense(decoder_outputs)
# 组建解码模型
decoder_model = Model(inputs=[decoder_inputs]+decoder_states_inputs,
                      outputs=[decoder_outputs]+decoder_states)

def decode_sequence(input_seq):
    # 得到输入英文被编码的状态
    states_value = encoder_model.predict(input_seq)
    #存放编码之后的法文的one-hot编码
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0,0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 获得one-hot的目标只
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 如果得到的One-hot编码的长度超过了设定的法文的最大长度，则停止循环
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1,1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # 输入的英文
    input_seq = encoder_input_data[seq_index:seq_index+1]
    # 翻译之后得到的法文
    decoded_sentence = decode_sequence(input_seq)
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

