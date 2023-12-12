# -*- coding:utf-8 -*-
# @Author  : Ricky Xu
# @Time    : 2023/8/21 4:39 下午
# @FileName: train.py
# @Software: PyCharm

# 根据四个pkl进行训练，预测得到results下的csv  (有四个的pkl后，可单独运行)

import tensorflow.compat.v1 as tf
from data_utils import BatchManager, get_dict
from model import Model
import time
import csv
import os
import shutil
import logging  # 添加日志模块

# 把日志保存到txt
# 配置日志文件的位置和日志级别
logging.basicConfig(filename='data/training_log.txt', level=logging.INFO)

batch_size = 20
dict_file = 'data/prepare/dict.pkl'
result_file = 'data/prepare/results'


def train():
    # ------------------------数据准备--------------------------
    train_manager = BatchManager(batch_size=20)
    test_manager = BatchManager(batch_size=100, name='test')
    predict_manager = BatchManager(batch_size=100, name='predict')

    # ------------------------读取字典--------------------------
    mapping_dict = get_dict(dict_file)

    # ------------------------搭建模型--------------------------
    model = Model(mapping_dict)
    rets10 = []
    pres10 = []

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1):  # 设置epoch！！！！
            j = 0
            for batch, batch_index in train_manager.iter_batch(shuffle=True):
                print_loss(model, sess, batch, True, train_manager, i, j)
                j += 1

        rets = []
        pres = []
        for batch, batch_index in test_manager.iter_batch(shuffle=True):
            print_loss(model, sess, batch, False, test_manager, i, ++j)
            ret = model.predict(sess, batch, batch_index)
            rets.append(ret)
            print(ret)
            j += 1

        for k in range(3):
            for batch, batch_index in predict_manager.iter_batch(shuffle=True):
                print_loss(model, sess, batch, False, predict_manager, i, ++j)
                ret = model.predict(sess, batch, batch_index)
                if k == 0:
                    pres.append(ret)
                else:
                    for i, r in enumerate(ret):
                      pres[0][i].insert(-1, r[1])
                print(ret)
                j += 1
        rets10.append(rets)
        pres10.append(pres)

    # ------------------------写文件--------------------------
    if os.path.exists('data/prepare/results'):
        shutil.rmtree('data/prepare/results')
    os.makedirs('data/prepare/results')
    write_csv(result_file + "/test", rets10)
    write_csv(result_file + "/predict", pres10)


def write_csv(filename, rets):
    for index, ret in enumerate(rets):
        with open(filename + str(index) + ".csv", 'w', newline='', encoding="utf_8_sig") as csvfile:
            csvwriter = csv.writer(csvfile)
            for rows in ret:
                for row in rows:
                    csvwriter.writerow(row)


def print_loss(model, sess, batch, istrain, manager, i, j):
    start = time.time()
    loss = model.run_step(sess, batch)  # 已经封装好
    end = time.time()
    t = "train "
    if not istrain:
        t = "test "
    if j % 10 == 0:
        log_message = t + 'epoch:{},step:{}/{},loss:{},elapse:{},estimate:{}'.format(i + 1,
                                                                                     j,
                                                                                     manager.len_data,
                                                                                     loss,
                                                                                     end - start,
                                                                                     (end - start) * (
                                                                                             manager.len_data - j))  # 第几轮 第几步多少个批次 loss是多少
        print(log_message)  # 打印日志信息
        logging.info(log_message)  # 将日志信息保存到文件


if __name__ == '__main__':
    train()
