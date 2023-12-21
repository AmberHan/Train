# -*- coding:utf-8 -*-
# @Author  : Ricky Xu
# @Time    : 2023/8/17 3:57 下午
# @FileName: data_utils.py
# @Software: PyCharm

# 根据csv生成test,train,predict的pkl
# 把拼出来的所有句子都合在一起


import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os
import random
import math
from read_pkl import *


def get_data_with_windows(name='train'):
    with open(f'data/prepare/dict.pkl', 'rb') as f:
        map_dict = pickle.load(f)

    def item2id(data, w2i):
        return [w2i[x] if x in w2i else w2i['UNK'] for x in data]

    results = []
    root = os.path.join('data/prepare', name)
    files = list(os.listdir(root))

    for file in tqdm(files):
        result = []
        path = os.path.join(root, file)
        samples = pd.read_csv(path, sep=',')
        num_samples = len(samples)
        sep_index = [-1] + samples[samples['word'] == 'sep'].index.tolist() + [
            num_samples]  # 结果应为如：-1 20 40 50,-1是因为多个sep

        # ---------------------------------获取句子并将句子全部都转换成id-------------------------------
        for i in range(len(sep_index) - 1):  # 遍历每一个下标
            start = sep_index[i] + 1  # 拿到了每一句的start和end
            end = sep_index[i + 1]
            data = []
            for feature in samples.columns:  # 做的时候同时转换成下标
                data.append(item2id(list(samples[feature])[start:end], map_dict[feature][1]))
            result.append(data)

        # ---------------------------------数据增强-------------------------------
        # 拼接的目的： 不拼、拼两个、拼三个可以增加学习长短句子的能力，即数据增强
        results.extend(result)
        if name == 'train':
            two = []
            for i in range(len(result) - 1):
                first = result[i]
                second = result[i + 1]
                two.append([first[k] + second[k] for k in range(len(first))])

            three = []
            for i in range(len(result) - 2):
                first = result[i]
                second = result[i + 1]
                third = result[i + 2]
                three.append([first[k] + second[k] + third[k] for k in range(len(first))])
            results.extend(two + three)

    with open(f'data/prepare/' + name + '.pkl', 'wb') as f:
        pickle.dump(results, f)


def get_dict(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict


class BatchManager(object):
    def __init__(self, batch_size, name='train'):
        with open(f'data/prepare/' + name + '.pkl', 'rb') as f:
            data = pickle.load(f)  # len表示有多少句话
        self.batch_data, self.batch_index = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))  # 共有多少个批次
        sorted_IndexData = sorted(enumerate(data), key=lambda x: len(x[1][0]))  # 按照句子长度排序
        sorted_data = [item[1] for item in sorted_IndexData]
        sorted_index = [item[0] for item in sorted_IndexData]
        batch_data, batch_index = list(), list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * int(batch_size):(i + 1) * int(batch_size)]))
            batch_index.append(sorted_index[i * int(batch_size):(i + 1) * int(batch_size)])
        return batch_data, batch_index

    @staticmethod
    def pad_data(data):
        chars = []
        bounds = []
        flags = []
        radicals = []
        pinyins = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            char, bound, flag, target, radical, pinyin = line
            padding = [0] * (max_length - len(char))
            chars.append(char + padding)
            bounds.append(bound + padding)
            flags.append(flag + padding)
            targets.append(target + padding)
            radicals.append(radical + padding)
            pinyins.append(pinyin + padding)
        return [chars, bounds, flags, radicals, pinyins, targets]

    def iter_batch(self, shuffle=False):
        combined = list(zip(self.batch_data, self.batch_index))
        if shuffle:
            random.shuffle(combined)
            self.batch_data, self.batch_index = zip(*combined)
        for idx in range(self.len_data):
            yield self.batch_data[idx], self.batch_index[idx]


def utilMain(onlypredict=False):
    if not onlypredict:
        get_data_with_windows('train')
        get_data_with_windows('test')
    else:
        get_data_with_windows('predict')
    pkls_to_txt(onlypredict)
    # train_data=BatchManager(10,'train')
    # train_data=BatchManager(10,'test')


if __name__ == '__main__':
    utilMain()
