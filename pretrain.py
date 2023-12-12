# -*- coding:utf-8 -*-
# @Author  : Ricky Xu
# @Time    : 2023/9/6 10:01 上午
# @FileName: pretrain.py
# @Software: PyCharm


import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification
from data_utils import BatchManager, get_dict

# import os
# os.environ['NO_PROXY'] = 'huggingface.co'
# os.environ['CURL_CA_BUNDLE'] = ''


# os.environ['REQUESTS_CA_BUNDLE'] = '/cacert.crt'

batch_size = 20
dict_file = 'data/prepare/dict.pkl'
bert_model_name = 'bert-base-chinese'
bert_model_name1 = "D:/desk/t/bert-base-chinese.tar.gz"


def train():
    # 数据准备
    train_manager = BatchManager(batch_size=batch_size, name='train')
    test_manager = BatchManager(batch_size=100, name='test')

    # 读取字典
    mapping_dict = get_dict(dict_file)

    # 加载BERT模型和Tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name1)
    model = TFBertForTokenClassification.from_pretrained(bert_model_name1, num_labels=len(mapping_dict['label'][0]))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    for epoch in range(10):
        for batch, batch_index in train_manager.iter_batch(shuffle=True):
            # 处理输入数据并生成模型所需的输入
            input_texts = batch[0]  # input_texts应该是一个字符串列表
            inputs = tokenizer(input_texts, return_tensors="tf", padding=True, truncation=True,
                               is_split_into_words=True)

            with tf.GradientTape() as tape:
                logits = model(inputs['input_ids'], training=True)[0]
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(batch[5], logits, from_logits=True))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

        for batch, batch_index in test_manager.iter_batch(shuffle=True):
            input_texts = batch[0]
            inputs = tokenizer(input_texts, return_tensors="tf", padding=True, truncation=True,
                               is_split_into_words=True)
            logits = model(inputs['input_ids'], training=False)[0]

            # 处理模型的输出，进行标签预测等


if __name__ == '__main__':
    train()
