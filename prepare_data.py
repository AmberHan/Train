# -*- coding:utf-8 -*-
# @Author  : Ricky Xu
# @Time    : 2023/8/11 10:23 上午
# @FileName: prepare_data.py
# 生成train和test目录下的csv以及dict.pkl（重新运行会删除prepare目录）
# @Software: PyCharm

import os
import pandas as pd
import pickle
from collections import Counter
from data_process import split_text
import jieba.posseg as psg
from cnradical import Radical, RunOption
import shutil
from random import shuffle
import tensorflow as tf

train_dir = 'data/train/20230810'
predict_dir = 'data/predict'

# 设置随机种子
tf.random.set_seed(17)


def is_Brackets(x):
    if x == '[' or x == ']' or x == '【' or x == '】':
        return True
    return False


def process_text(idx, path, dir, split_method=None):
    """
    读取文本 切割 然后打上标记 并提取词边界、词性、偏旁部首、拼音等文本特征
    :param idx: 文件名字 不含扩展名
    :param split_method: 切割文本的方法 是一个函数
    :param split_name: 最终保存的文件夹名字
    :return:
    """
    data = {}

    # -------------------------获取句子---------------------------
    if split_method is None:
        with open(f'{dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        with open(f'{dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = split_method(texts)
    data['word'] = texts

    # -------------------------获取标签----------------------------
    tag_list = ['O' for s in texts for x in s]
    # 读取这个文件对应的ann文件
    if dir != predict_dir:
        tag = pd.read_csv(f'{dir}/{idx}.ann', header=None, sep='\t')
        for i in range(tag.shape[0]):
            tag_item = tag.iloc[i][1].split(' ')  # 对获取对实体类别以及起始位置
            cls, start, end = tag_item[0], int(tag_item[1]), int(tag_item[-1])  # 转换成对应的类型
            tag_list[start] = 'B-' + cls  # 起始位置写入B-实体类别
            for j in range(start + 1, end):  # 后面的位置写I-实体类别
                tag_list[j] = 'I-' + cls
        assert len([x for s in texts for x in s]) == len(tag_list)  # 保证两个序列长度一直

    # -------------------------提取词性和词边界特征-------------------------
    word_bounds = ['M' for item in tag_list]  # word_bounds保留分词的边界，首先给所有的字都标上B标记。B-begin M-mid E-end S-单独的字
    word_flags = []  # word_flags保存词性特征，ns-国家名字，n-名词，m-数字...
    for text in texts:
        for word, flag in psg.cut(text):
            if len(word) == 1:  # 判断是一个字的词
                start = len(word_flags)  # 拿到起始下标
                word_bounds[start] = 'S'  # 标记修改为S
                word_flags.append(flag)  # 将当前词的词性名加入到wordflags列表
            else:
                start = len(word_flags)  # 获取起始下标
                word_bounds[start] = 'B'  # 第一个字打上B
                word_flags += [flag] * len(word)  # 将这个词的每个字都加上词性标记
                end = len(word_flags) - 1  # 拿到这个词的最后一个字的下标
                word_bounds[end] = 'E'  # 将最后一个字打上E标记

    # ----------------------------统一截断------------------------------
    tags = []
    bounds = []
    flags = []
    start = 0
    end = 0
    for s in texts:
        l = len(s)
        end += l
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        tags.append(tag_list[start:end])
        start += l
    data['bound'] = bounds
    data['flag'] = flags
    data['label'] = tags
    # 将flag列内的所有内容替换为"abc"
    # data['flag'] = [['abc'] * len(s) for s in data['flag']]
    # 将bound列内的所有内容替换为"abc"
    # data['bound'] = [['abc'] * len(s) for s in data['bound']]

    # ----------------------获取拼音特征---------------------------
    radical = Radical(RunOption.Radical)  # 提取偏旁部首
    pinyin = Radical(RunOption.Pinyin)  # 用来提取拼音
    # 提取偏旁部首特征，对于没有偏旁部首的字标上PAD
    data['radical'] = [
        [radical.trans_ch(x) if radical.trans_ch(x) is not None else 'KUO' if is_Brackets(x) else 'UNK' for x in s]
        for s in texts]
    # 将radical列内的所有内容替换为"abc"
    # data['radical'] = [['abc'] * len(s) for s in data['radical']]

    # 提取拼音特征，对于没有拼音的字标上PAD
    data['pinyin'] = [
        [pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'KUO' if is_Brackets(x) else 'UNK' for x in s] for
        s in
        texts]  # 把标点替换成PAD
    # 将pinyin列内的所有内容替换为"abc"
    # data['pinyin'] = [['abc'] * len(s) for s in data['pinyin']]

    # ------------------------存储数据-----------------------------
    num_samples = len(texts)  # 获取有多少句话 等于是有多少个样本
    num_col = len(data.keys())  # 获取特征的个数 也就是列数

    dataset = []
    for i in range(num_samples):
        records = list(zip(*[list(v[i]) for v in data.values()]))  # 解压
        dataset += records + [['sep'] * num_col]  # 每存完一个句子需要一行sep进行隔离
    dataset = dataset[:-1]  # 最后一行sep不要
    dataset = pd.DataFrame(dataset, columns=data.keys())  # 转化为dataframe
    save_path = f'{path}/{idx}.csv'

    def clean_word(w):
        if w == '\n':
            return 'LB'
        if w in [' ', '\t', '\u2003']:
            return 'SPACE'
        if w.isdigit():  # 将所有的数字都变成一种符号
            return 'num'
        return w

    dataset['word'] = dataset['word'].apply(clean_word)
    dataset.to_csv(save_path, index=False, encoding='utf-8')


def multi_process(split_method=None, onlyPredict=False, train_radio=0.8):  # 0.8来做训练
    train_folder = 'data/prepare/train'
    test_folder = 'data/prepare/test'
    pre_folder = 'data/prepare/predict'
    import multiprocessing as mp
    num_cpus = mp.cpu_count()  # 获取机器cpu的个数
    my_use = num_cpus // 2
    print("CPU核数为:" + str(num_cpus) + ",我使用" + str(my_use))
    pool = mp.Pool(my_use)
    results = []
    if onlyPredict:
        if os.path.exists(pre_folder):
            shutil.rmtree(pre_folder)
            os.makedirs(pre_folder)
        pre_idxs = list(
            set([file.split('.')[0] for file in os.listdir(predict_dir) if
                 file.endswith('.txt')]))
        for idx in pre_idxs:
            result = pool.apply_async(process_text, args=(idx, pre_folder, predict_dir, split_method))
            results.append(result)
    else:
        if os.path.exists('data/prepare/'):
            shutil.rmtree('data/prepare/')
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
            os.makedirs(test_folder)
            os.makedirs(pre_folder)
        idxs = list(
            set([file.split('.')[0] for file in os.listdir(train_dir) if
                 file.endswith('.txt')]))  # 获取所有文件名字
        pre_idxs = list(
            set([file.split('.')[0] for file in os.listdir(predict_dir) if
                 file.endswith('.txt')]))
        shuffle(idxs)  # 打乱顺序
        index = int(len(idxs) * train_radio)  # 拿到训练集的截止下标
        train_ids = idxs[:index]  # 训练集文件名集合
        test_ids = idxs[index:]  # 测试集文件名集合

        for idx in train_ids:
            result = pool.apply_async(process_text, args=(idx, train_folder, train_dir, split_method))
            results.append(result)

        for idx in test_ids:
            result = pool.apply_async(process_text, args=(idx, test_folder, train_dir, split_method))
            results.append(result)

        for idx in pre_idxs:
            result = pool.apply_async(process_text, args=(idx, pre_folder, predict_dir, split_method))
            results.append(result)

    pool.close()
    pool.join()
    [r.get() for r in results]


def mapping(data, threshold=10, is_word=False, sep='sep', is_label=False):
    count = Counter(data)
    if sep is not None and count[sep] != 0:
        count.pop(sep)
    if is_word:
        #   ##########
        #   ######PPPP
        #   ########PP
        #   ####PPPPPP
        # 然后 句子要按长度排序。   why？    因为每个批次数据在进行填充的时候是以本批次中最长的那个句子作为标准的
        # 句子长度差不多的句子会在一起 这个时候再去分批次，每个数据的长度就是接近的，提高运算速度
        count['PAD'] = 100000001  # PAD后面做填充用
        count['UNK'] = 100000000
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data if x[1] >= threshold]  # 去掉频率小于threshold的元素
        id2item = data  # 构造一个列表
        item2id = {id2item[i]: i for i in range(len(id2item))}  # 字典
    elif is_label:  # 是label的话就不能加pad
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    else:
        count['PAD'] = 100000001
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    return id2item, item2id


def get_dict():
    map_dict = {}
    from glob import glob
    all_w = []
    all_bound = []
    all_flag = []
    all_label = []
    all_radical = []
    all_pinyin = []
    for file in glob('data/prepare/train/*.csv'):  # + glob('data/prepare/test/*.csv'):  # 实际上这里不应该放测试集
        df = pd.read_csv(file, sep=',')
        all_w += df['word'].tolist()
        all_bound += df['bound'].tolist()
        all_flag += df['flag'].tolist()
        all_label += df['label'].tolist()
        all_radical += df['radical'].tolist()
        all_pinyin += df['pinyin'].tolist()
    map_dict['word'] = mapping(all_w, threshold=1, is_word=True)
    map_dict['bound'] = mapping(all_bound)
    map_dict['flag'] = mapping(all_flag)
    map_dict['label'] = mapping(all_label, is_label=True)
    map_dict['radical'] = mapping(all_radical)
    map_dict['pinyin'] = mapping(all_pinyin)
    with open(f'data/prepare/dict.pkl', 'wb') as f:
        pickle.dump(map_dict, f)


def dataMain(onlypredict=False):
    # print(process_text('003', split_method=split_text,split_name='train'))
    #  multi_process()
    # print(set([file.split('.')[0] for file in os.listdir(train_dir)]))
    multi_process(split_text, onlypredict)
    if not onlypredict:
        get_dict()
    # with open(f'data/prepare/dict.pk1','rb') as f:
    #     data=pickle.load(f)
    # print(data['bound'])


if __name__ == '__main__':
    dataMain()
