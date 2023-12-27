# -*- coding:utf-8 -*-
# @Author  : Ricky Xu
# @Time    : 2023/8/18 3:07 下午
# @FileName: model.py
# @Software: PyCharm

import tensorflow.compat.v1 as tf
import numpy as np

# 设置全局随机种子
# tf.set_random_seed(17)


# from tensorflow.contrib import rnn
from tensorflow_addons.text.crf import crf_log_likelihood, viterbi_decode


def network(inputs, shapes, num_tags, lstm_dim=100, initializer=tf.random_normal_initializer()):
    # def network(inputs,shapes,num_tags,lstm_dim=100, initializer=tf.truncated_normal_initializer()):
    """
    接受一个批次样本的特征数据，计算出网络的输出值
    :param char: type of int, id of chars, a tensor of shape 2-D[None.None]
    :param bound: a tensor of shape 2-D[None.None] with type of int
    :param flag: a tensor of shape 2-D[None.None] with type of int
    :param radical: a tensor of shape 2-D[None.None] with type of int
    :param pinyin: a tensor of shape 2-D[None.None] with type of int
    :return: 直接返回
    """
    # ----------------------------特征嵌入-------------------------------------
    # 将所有特征的id转换成一个固定长度的向量
    embedding = []
    keys = list(shapes.keys())
    for key in keys:
        with tf.variable_scope(key + '_embedding', reuse=tf.AUTO_REUSE):
            lookup = tf.get_variable(
                name=key + '_embedding',
                shape=shapes[key],
                initializer=initializer
            )
            # bound的映射
            embedding.append(tf.nn.embedding_lookup(lookup, inputs[key]))  # 实现特征的嵌入
    embed = tf.concat(embedding, axis=-1)  # shape [None,None,char_dim+bound_dim+flag_dim+radical_dim+pinyin_dim]

    sign = tf.sign(tf.abs(inputs[keys[0]]))  # 每一行求和就得到了每个句子的真实长度
    lengths = tf.reduce_sum(sign, reduction_indices=1)  # 求出每一个句子的真实长度，每一行求和就得到了每个句子的真实长度
    num_time = tf.shape(inputs[keys[0]])[1]

    # --------------------------循环神经网络编码---------------------------------
    with tf.variable_scope('BiLstm_layer1', reuse=tf.AUTO_REUSE):  # 双层双向循环神经网络
        lstm_cell = {}
        for name in ['forward', 'backward']:
            with tf.variable_scope(name):
                lstm_cell[name] = tf.nn.rnn_cell.BasicLSTMCell(
                    lstm_dim  # 初始化
                )
        outputs, final_states1 = tf.nn.bidirectional_dynamic_rnn(  # 双向动态rnn
            lstm_cell['forward'],
            lstm_cell['backward'],
            embed,
            dtype=tf.float32,
            sequence_length=lengths
        )
    output1 = tf.concat(outputs, axis=-1)  # b, L, 2*lstm_dim

    with tf.variable_scope('BiLstm_layer2', reuse=tf.AUTO_REUSE):  # 双层双向循环神经网络
        lstm_cell = {}
        for name in ['forward', 'backward']:
            with tf.variable_scope(name):
                lstm_cell[name] = tf.nn.rnn_cell.BasicLSTMCell(
                    lstm_dim  # 初始化
                )
        outputs, final_states1 = tf.nn.bidirectional_dynamic_rnn(  # 双向动态rnn
            lstm_cell['forward'],
            lstm_cell['backward'],
            output1,
            dtype=tf.float32,
            sequence_length=lengths
        )
    output = tf.concat(outputs, axis=-1)  # batch_size, maxlength, 2*lstm_dim

    # --------------------------输出映射---------------------------------
    output = tf.reshape(output, [-1, 2 * lstm_dim])  # reshape成二维矩阵   batch_size*maxlength,2*lstm_dim
    with tf.variable_scope('project_layer1'):
        w = tf.get_variable(
            name='w',
            shape=[2 * lstm_dim, lstm_dim],
            initializer=initializer
        )
        b = tf.get_variable(
            name='b',
            shape=[lstm_dim],
            initializer=tf.zeros_initializer()

        )
        output = tf.nn.relu(tf.matmul(output, w) + b)

    with tf.variable_scope('project_layer2'):
        w = tf.get_variable(
            name='w',
            shape=[lstm_dim, num_tags],
            initializer=initializer
        )
        b = tf.get_variable(
            name='b',
            shape=[num_tags],
            initializer=tf.zeros_initializer()

        )
        output = tf.matmul(output, w) + b
    output = tf.reshape(output, [-1, num_time, num_tags])

    return output, lengths  # batch_size, max_length, num_tags


# 训练的部分不需要model来实现，model是为了从你的输入到输出，计算，损失，优化器
class Model(object):

    def __init__(self, dict, lr=0.0001, reuse=None):  # 设置学习率！！！！！
        # lr=0.0001: epoch=10,f1=66; epoch=20,f1=70; epoch=30,f1=74.7
        # lr=0.01: epoch=10, f1=90
        # lr=0.005:epoch=30, f1=87.8######## epoch=25 f1=88.76 epoch=20 f1=89。8  epoch=15 f1=84。3
        # lr=0.0025:epoch=10,f1=80.7
        # lr=0.004:epoch=10, f1=84.7
        # lr=0.0005:epoch=30, f1=81.5
        # lr=0.0001:epoch=30, f1=80
        # --------------------------用到的参数值---------------------------
        self.num_char = len(dict['word'][0])
        self.num_bound = len(dict['bound'][0])
        self.num_flag = len(dict['flag'][0])
        self.num_radical = len(dict['radical'][0])
        self.num_pinyin = len(dict['pinyin'][0])
        self.num_tags = len(dict['label'][0])
        self.char_dim = 100
        self.bound_dim = 20
        self.flag_dim = 50
        self.radical_dim = 50
        self.pinyin_dim = 50
        self.lstm_dim = 100
        self.lr = lr  # 学习率！！！！！！！！！！
        self.map = dict

        # --------------------------定义接收数据的placeholder---------------------------
        tf.compat.v1.disable_eager_execution()
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='char_inputs')
        self.bound_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='bound_inputs')
        self.flag_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='flag_inputs')
        self.radical_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='radical_inputs')
        self.pinyin_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pinyin_inputs')
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
        self.global_step = tf.Variable(0, trainable=False)  # 不需要训练，只用来计数

        # ---------------------------计算模型输出值------------------------------------------------
        self.logits, self.lengths = self.get_logits(self.char_inputs,
                                                    self.bound_inputs,
                                                    self.flag_inputs,
                                                    self.radical_inputs,
                                                    self.pinyin_inputs)

        # ------------------------------计算损失---------------------------------------------------
        self.cost = self.loss(self.logits, self.targets, self.lengths)

        # ---------------------------------优化器优化----------------------------------------------
        # 采用梯度截断技术
        with tf.variable_scope('optimizer'):
            opt = tf.train.AdamOptimizer(self.lr)
            grad_vars = opt.compute_gradients(self.cost)  # 计算出所有参数的导数
            clip_grad_vars = [[tf.clip_by_value(g, -5, 5), v] for g, v in
                              grad_vars]  # 按照值来截断导数（损失值），得到截断之后的梯度。限制导数，导数的绝对值过大会导致梯度爆炸
            self.train_op = opt.apply_gradients(clip_grad_vars, self.global_step)  # 使用截断后的梯度对参数进行更新

    # 做预测的时候只要网络的输出值就可以，不需要计算损失
    def get_logits(self, char, bound, flag, radical, pinyin):
        """
        接受一个批次样本的特征数据，计算出网络的输出值
        :param char: type of int, id of chars, a tensor of shape 2-D[None.None]
        :param bound: a tensor of shape 2-D[None.None] with type of int
        :param flag: a tensor of shape 2-D[None.None] with type of int
        :param radical: a tensor of shape 2-D[None.None] with type of int
        :param pinyin: a tensor of shape 2-D[None.None] with type of int
        :return: 3-d tensor [batch_size, max_length, num_tags]
        """
        shapes = {}
        shapes['char'] = [self.num_char, self.char_dim]
        shapes['bound'] = [self.num_bound, self.bound_dim]
        shapes['flag'] = [self.num_flag, self.flag_dim]
        shapes['radical'] = [self.num_radical, self.radical_dim]
        shapes['pinyin'] = [self.num_pinyin, self.pinyin_dim]
        inputs = {}
        inputs['char'] = char
        inputs['bound'] = bound
        inputs['flag'] = flag
        inputs['radical'] = radical
        inputs['pinyin'] = pinyin

        return network(inputs, shapes, lstm_dim=self.lstm_dim, num_tags=self.num_tags)

    # 将原本条件随机场中用来提取特征的部分，在本来的crf++算法中用到是传统的统计方法去统计出每一个时刻每一种状态的分值，
    # 但在这里每一个时刻31个状态的分值我们是用模型（network）来完成，因此该模型被称为"Bilstm+crf"
    # Bilstm负责提取特征   crf是最终模型
    def loss(self, output, targets, lengths):
        b = tf.shape(lengths)[0]
        num_steps = tf.shape(output)[1]
        with tf.variable_scope('crf_loss'):
            small = -1000.0
            start_logits = tf.concat(
                [small * tf.ones(shape=[b, 1, self.num_tags], dtype=tf.float32),
                 tf.zeros(shape=[b, 1, 1], dtype=tf.float32)], axis=-1
            )
            pad_logits = tf.cast(small * tf.ones([b, num_steps, 1]), tf.float32)
            logits = tf.concat([output, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([b, 1]), tf.int32), targets], axis=-1
            )
            self.trans = tf.get_variable(
                name='train',
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=tf.truncated_normal_initializer()
            )
            log_likehood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths
            )
            return tf.reduce_mean(-log_likehood)

    def run_step(self, sess, batch, istrain=True):
        if istrain:  # 训练
            feed_dict = {
                self.char_inputs: batch[0],
                self.bound_inputs: batch[1],
                self.flag_inputs: batch[2],
                self.radical_inputs: batch[3],
                self.pinyin_inputs: batch[4],
                self.targets: batch[5]
            }
            _, loss = sess.run([self.train_op, self.cost], feed_dict=feed_dict)
            return loss
        else:  # 测试
            feed_dict = {
                self.char_inputs: batch[0],
                self.bound_inputs: batch[1],
                self.flag_inputs: batch[2],
                self.radical_inputs: batch[3],
                self.pinyin_inputs: batch[4]
            }
            logits, lengths = sess.run([self.logits, self.lengths], feed_dict=feed_dict)
            return logits, lengths

    def decode(self, logits, lengths, matrix):
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]  # 只取有效字符的输出
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=-1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            paths.append(path[1:])
        return paths

    def predict(self, sess, batch, batch_index):  # 完成预测
        results = []
        matrix = self.trans.eval()  # 拿到转移矩阵
        logits, lengths = self.run_step(sess, batch, istrain=False)
        paths = self.decode(logits, lengths, matrix)
        chars = batch[0]
        labels = batch[-1]
        for i in range(len(paths)):
            length = lengths[i]
            string = [self.map['word'][0][index] for index in chars[i][:length]]
            real = [self.map['label'][0][index] for index in labels[i][:length]]
            tags = [self.map['label'][0][index] for index in paths[i]]
            index = [batch_index[i]] * len(paths[i])
            result = [k for k in zip(string, tags, real, index)]
            result_as_list = [list(item) for item in result]
            results.extend(result_as_list)
        return results
