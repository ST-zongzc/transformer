import os
import torch
from d2l import torch as d2l


def read_data_cmn_eng():
    """载入“英语－中文”数据集"""
    with open('data/cmn-eng/cmn_data_process.txt', 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_cmne_eng(text):
    """预处理“英语－中文”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?。，！？') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_cmne_eng(text, num_examples=None):
    """词元化“英语－中文”数据数据集"""
    source, target = [], []  # source 是源语言  target 是目标语言
    for i, line in enumerate(text.split('\n')):  # 按行遍历
        if num_examples and i > num_examples:  # 限制句子数
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))  # 分割成词元列表
            target_tmp = list(parts[1].split(' ')[0])
            target_tmp.append(parts[1].split(' ')[-1])
            target.append(target_tmp)
    return source, target


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize(figsize=(8, 4))
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
    d2l.plt.show()


if __name__ == '__main__':
    raw_text = read_data_cmn_eng()
    print(raw_text[:67])

    print("===================")

    text = preprocess_cmne_eng(raw_text)
    print(text[:75])

    print("===================")

    source, target = tokenize_cmne_eng(text)
    print(source[:6])
    print(target[:6])

    print("===================")

    show_list_len_pair_hist(['source', 'target'],
                            '# tokens per sequence',
                            'count', source, target)

    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(len(src_vocab))
