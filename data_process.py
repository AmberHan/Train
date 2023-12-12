# -*- coding:utf-8 -*-
# @Author  : Ricky Xu
# @Time    : 2023/8/10 9:52 上午
# @FileName: data_process.py
# @Software: PyCharm


import os
import re

train_dir = 'data/train/20230810'


def get_entities(dir):
    entities = {}  # 用来存储实体
    files = os.listdir(dir)
    files = list(set([file.split('.')[0] for file in files]))
    for file in files:
        path = os.path.join(dir, file + '.ann')
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                name = line.split('\t')[1].split(' ')[0]
                if name in entities:
                    entities[name] += 1
                else:
                    entities[name] = 1
    return entities


# 功能是得到标签和下标的映射
def get_labelencoder(entities):
    entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
    entities = [x[0] for x in entities]
    id2label = []
    id2label.append('0')
    for entity in entities:
        id2label.append('B-' + entity)
        id2label.append('I-' + entity)
    label2id = {id2label[i]: i for i in range(len(id2label))}
    return id2label, label2id


# 数据预处理——按标点符号断开
def split_text(text):
    split_index = []  # 保存每次切出来的结果

    pattern1 = '\.|,|，|:|;|\.|\\?|。|；|：|？'
    for m in re.finditer(pattern1, text):
        idx = m.span()[0]  # 找到标点下标
        # 考虑各种具体情况是否要切分
        if text[idx - 1] == '\n':  # 前换行 后标点
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isdigit():  # 前数字 后数字
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isspace() and text[idx + 2].isdigit():  # 前数字 后空格
            continue
        if text[idx + 1] in set('.。;；,，'):  # 两个标点连着
            continue
        split_index.append(idx + 1)
    pattern2 = '第[一二三四五六七八九零十]出'  # 应在前面切
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        if (text[idx:idx + 2] in ['or', 'by'] or text[idx:idx + 3] == 'and' or text[idx:idx + 4] == 'with') \
                and (text[idx - 1].islower() or text[idx - 1].isupper()):
            continue
        split_index.append(idx)
    pattern3 = '\n\d\.'
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        if ischinese(text[idx + 3]):
            split_index.append(idx)
    for m in re.finditer('\n\(\d\)', text):
        idx = m.span()[0]
        split_index.append(idx + 1)
    split_index = list(sorted(set([0, len(text)] + split_index)))
    ret = []
    for i in range(len(split_index) - 1):
        print(i, '||||', text[split_index[i]:split_index[i + 1]])
        ret.append(text[split_index[i]:split_index[i + 1]])
    return ret


# 判断是不是中文
def ischinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False


def myMain():
    # # print(get_entities(train_dir))  # 各实体种类个数
    # # print(len(get_entities(train_dir)))  # 总实体种类数，需要bio标注的数量=总实体种类数*2+1
    # #
    # # entities = get_entities(train_dir)
    # # label = get_labelencoder(entities)
    # # print(label)
    # #
    # # pattern = '.|,|，|:|;|\.|?|。|；|：|？'
    # # with open('data/train/20230810/003.txt', 'r', encoding='utf8') as f:
    # #     text = f.read()
    # #     for m in re.finditer(pattern, text):
    # #         print(m)
    #
    #
    files = os.listdir(train_dir)
    files = list(set([file.split('.')[0] for file in files if file.split('.')[0] != '']))
    # pattern1 = '\.|,|，|:|;|\.|\\?|。|；|：|？'
    # pattern2 = '第[一二三四五六七八九零十]出'
    # for file in files:
    #     path=os.path.join(train_dir,file+'.txt')
    #     with open(path, 'r', encoding='utf8') as f:
    #         text=f.read()
    #         for m in re.finditer(pattern2, text):
    #             idx = m.span()[0]  # 找到标点下标
    #             print(file+'||||', text[idx-40:idx+40])
    # print('finish')

    path = os.path.join(train_dir, files[0] + '.txt')
    # path = 'data/train/20230810/汤显祖南柯记.txt'
    with open(path, 'r', encoding='utf8') as f:
        text = f.read()
        split_text(text)


if __name__ == '__main__':
    myMain()
