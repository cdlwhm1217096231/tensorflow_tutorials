#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-13 08:34:02
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from __future__ import absolute_import, print_function, division

import os
import zipfile
import random
import urllib
import collections

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

# 训练参数
learning_rate = 0.1
batch_size = 128
num_steps = 3000000   # 总共训练的次数3000000
display_step = 10000   # 每隔10000次，打印一次输出的结果
eval_step = 200000  # 每隔200000次，评估模型输出一次评估结果

# 评价参数
eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']
# Word2Vec参数
embedding_size = 200  # 词向量的维度
max_vocabulary_size = 50000  # 词汇表中不同单词的总数
min_occurrence = 10  # 删除出现次数少于10次的单词，即删除生僻的单词
skip_window = 3  # 窗口的大小
num_skips = 2   # 获取两组训练样本
num_sampled = 64  # 负采样的样本数量
# 下载数据集
url = "http://mattmahoney.net/dc/text8.zip"
data_path = "./dataset/text8.zip"
if not os.path.exists(data_path):
    print("Downloading the dataset.....")
    filename, _ = urllib.request.urlretrieve(url, data_path)
    print("下载完成!")

# 解压文件
with zipfile.ZipFile(data_path) as f:
    text_words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    # text_words = (f.read(f.namelist()[0]).lower().decode('utf-8')).split()
# 构建词汇表(词汇表中的每个单词都是独一无二的),使用UNK标识符代替生僻单词
count = [("UNK", -1)]
# 检索出最常见的单词
count.extend(collections.Counter(
    text_words).most_common(max_vocabulary_size - 1))
# 删除出现次数小于min_occurrence的单词
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        break
# 计算词汇表的大小
vocabulary_size = len(count)
# 给每个单词分配对应的索引
word2id = dict()  # word2id字典中，字典的key是单词的名称，value是单词在词汇表中的索引
for i, (word, _) in enumerate(count):
    word2id[word] = i

data = list()  # 存放单词对应索引的列表
unk_count = 0
for word in text_words:
    # 检索一个单词的索引，如果单词没有出现的词汇表中，则称为是UNK,此时对应的索引的0
    index = word2id.get(word, 0)  # 返回字典指定key所对应的字典value值,当单词没有出现在字典的key中时，返回0
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ("UNK", unk_count)
# id2word字典中，字典的key是单词在字典中的索引，字典的value是单词的名称
id2word = dict(zip(word2id.values(), word2id.keys()))
print("Words count:", len(text_words))   # 原始语料库中的词汇数量
print("Unique Words:", len(set(text_words)))  # 语料库中独一无二的词汇数量
print("vocabulary size:", vocabulary_size)  # 设置的one-hot向量的维度
print("Most common words:", count[:10])

data_index = 0


# 为skip-gram模型生成training batch
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 获得上下文词汇的范围
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index: data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# Input data
X = tf.placeholder(tf.int32, shape=[None])
# Input labels
Y = tf.placeholder(tf.int32, shape=[None, 1])

with tf.device("/cpu:0"):
    # create the embedding variable(每行代表一个单词的词向量)
    embedding = tf.Variable(tf.random_normal(
        [vocabulary_size, embedding_size]))
    # lookup将每个样本X与词向量对应
    X_embed = tf.nn.embedding_lookup(embedding, X)
    # construct the variable for NCE loss
    nce_weights = tf.Variable(tf.random_normal(
        [vocabulary_size, embedding_size]))
    nce_bias = tf.Variable(tf.zeros([vocabulary_size]))

# compute the nce loss for the batch
loss_op = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_bias,
                   labels=Y, inputs=X_embed,
                   num_sampled=num_sampled,
                   num_classes=vocabulary_size))

# define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)


# 评估模型
# 计算余弦相似度
X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))
embedding_norm = embedding / \
    tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
cosine_sim_op = tf.matmul(X_embed_norm, embedding_norm, transpose_b=True)


# initialize the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    x_test = np.array([word2id[w] for w in eval_words])
    average_loss = 0
    for step in range(1, num_steps + 1):
        # get a new batch of data
        batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
        # run training op
        _, loss = sess.run([train_op, loss_op], feed_dict={
                           X: batch_x, Y: batch_y})
        average_loss += loss

        if step % display_step == 0 or step == 1:
            if step > 1:
                average_loss /= display_step
            print("Step " + str(step) + ", Average Loss= " +
                  "{:.4f}".format(average_loss))
            average_loss = 0

        # 评估
        if step % eval_step == 0 or step == 1:
            print("Evaluation Model......")
            sim = sess.run(cosine_sim_op, feed_dict={X: x_test})
            for i in range(len(eval_words)):
                top_k = 8  # 找出与输入余弦相似度最高的8个单词
                nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                log_str = '"%s" nearest neighbors:' % eval_words[i]
                for k in range(top_k):
                    log_str = '%s %s,' % (log_str, id2word[nearest[k]])
                print(log_str)
    final_embeddings = embedding_norm.eval()

# 可视化word2vec效果


def plot_with_labels(low_dim_embs, labels, filename="tsne.png"):
    assert low_dim_embs.shape[0] >= len(labels)
    plt.figure(figsize=(20, 20))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom"
        )
        plt.savefig(filename)


tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [id2word[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
