import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# S: 起始标记
# E: 结束标记
# P：意为padding，将当前序列补齐至最长序列长度的占位符
sentence = [
    # encode_input   decode_input    decode_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E'],
]

# 词典，padding用0来表示
# 源词典
src_vocabulary = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocabulary_size = len(src_vocabulary)  # 6
# 目标词典（包含特殊符）
target_vocabulary = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
# 反向映射词典，idx ——> word
idx2word = {v: k for k, v in target_vocabulary.items()}
target_vocabulary_size = len(target_vocabulary)  # 9

src_len = 5  # 输入序列encode_input的最长序列长度，其实就是最长的那句话的token数
tgt_len = 6  # 输出序列decode_input/decode_output的最长序列长度


# 这个函数把原始输入序列转换成token表示
def make_data(sentence):
    encode_inputs, decode_inputs, decode_outputs = [], [], []
    for i in range(len(sentence)):
        encode_input = [src_vocabulary[word] for word in sentence[i][0].split()]
        decode_input = [target_vocabulary[word] for word in sentence[i][1].split()]
        decode_output = [target_vocabulary[word] for word in sentence[i][2].split()]

        encode_inputs.append(encode_input)
        decode_inputs.append(decode_input)
        decode_outputs.append(decode_output)

    # LongTensor是专用于存储整型的，Tensor则可以存浮点、整数、bool等多种类型
    return torch.LongTensor(encode_inputs), torch.LongTensor(decode_inputs), torch.LongTensor(decode_outputs)


encode_inputs, decode_inputs, decode_outputs = make_data(sentence)

print(' encode_inputs: \n', encode_inputs)  # enc_inputs: [2,5]
print(' decode_inputs: \n', decode_inputs)  # dec_inputs: [2,6]
print(' decode_outputs: \n', decode_outputs)  # dec_outputs: [2,6]


# 使用Dataset加载数据
class MyDataSet(Data.Dataset):
    def __init__(self, encode_inputs, decode_inputs, decode_outputs):
        super(MyDataSet, self).__init__()
        self.encode_inputs = encode_inputs
        self.decode_inputs = decode_inputs
        self.decode_outputs = decode_outputs

    def __len__(self):
        # 我们前面的encode_inputs.shape = [2,5],所以这个返回的是2
        return self.encode_inputs.shape[0]

        # 根据idx返回的是一组 encode_input, decode_input, decode_output

    def __getitem__(self, idx):
        return self.encode_inputs[idx], self.decode_inputs[idx], self.decode_outputs[idx]


# 构建DataLoader
loader = Data.DataLoader(dataset=MyDataSet(encode_inputs, decode_inputs, decode_outputs),
                         batch_size=2, shuffle=True)

# 用来表示一个词的向量长度
d_model = 512

# FFN的隐藏层神经元个数
d_ff = 2048

# 分头后的q、k、v词向量长度，依照原文我们都设为64
# 原文：queries and kes of dimention d_k,and values of dimension d_v .所以q和k的长度都用d_k来表示
d_k = d_v = 64

# Encoder Layer 和 Decoder Layer的个数
n_layers = 6

# 多头注意力中head的个数，原文：we employ h = 8 parallel attention layers, or heads
n_heads = 8


# Transformer 包含 Encoder 和 Decoder
# Encoder 和 Decoder 各自包含6个Layer
# Encoder Layer 中包含 多头自注意力 Multi-Head Self Attention 和 基于位置的前馈网络 Position-wise Feed-Forward Networks 两个sub Layer

# Decoder Layer 中包含 Masked Multi-Head Self Attention、 Cross Attention、 Position-wise Feed-Forward Networks 三个Sub Layer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # dropout是原文的0.1，max_len原文没找到
        """

        :param d_model:
        :param dropout:
        :param max_len: max_len是假设的一个句子最多包含 5000个token,即 max_seq_len
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 开始位置编码部分,首先生成一个max_len * d_model 的tensor，即5000 * 512
        # 5000是一个句子中最多的token数，512是一个token用多长的向量来表示，5000*512这个矩阵用于表示一个句子的信息
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # pos的shape为[max_len,1],即[5000,1]
        # 先把括号内的分式求出来,pos是[5000,1],分母是[256],通过广播机制相乘后是[5000,256]
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        # 再取正余弦
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        # 一个句子要做一次pe，一个batch中会有多个句子，所以增加一维来用和输入的一个batch的数据相加时做广播
        pe = pe.unsqueeze(0)  # [5000,512] -> [1,5000,512]
        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        :param x: [batch_size, seq_len, d_model]
        :return:
        """
        # 5000是我们预定义的最大的seq_len，就是说我们把最多的情况pe都算好了，用的时候用多少就取多少
        x = x + self.pe[:, :x.size(1), :]  # 加的时候应该也广播了，第一维 1 -> batch_size
        return self.dropout(x)  # return: [batch_size, seq_len, d_model], 和输入的形状相同
