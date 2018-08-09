#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.3.3

__date__ = '2018/7/30 15:08'
__author__ = 'cdl'

import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8
# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 在shape的一个维度上使用None可以方便使用不同的batch大小。在训练时，需要把数据分成比较小的batch；在测试时，可以一次性使用全部的数据。
# 当数据集比较小时，这样比较方便测试；但数据集比较大时，将大量的数据放在一个batch中可能会导致内存溢出。
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
# 定义神经网络的前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 定义损失函数和反向传播算法
y = tf.sigmoid(y)
# y_ 代表正确结果   y 代表预测结果   tf.clip_by_value（）可以将一个张量中的数值限制在一个范围内，这样可以避免被一些运算错误；大于1.0的数被换成1.0，小于1e-10的数被换成1e-10，保证log运算不会出现log0的错误

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
training_rate = 0.001  # 学习率
train_step = tf.train.AdamOptimizer(training_rate).minimize(cross_entropy)
# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]
# 创建一个会话运行tensorflow程序
with tf.Session() as sess:
    #  初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print('未经训练的神经网络参数w1:\n', sess.run(w1))
    print('未经训练的神经网络参数w2:\n', sess.run(w2))
    # 设置训练的次数
    STEPS = 5000
    for i in range(STEPS):
    # 每次选择batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
    # 通过选择的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i%100 == 0:
    #  每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print('经过%d次训练后，这个数据集的交叉熵是%g' % (i, total_cross_entropy))
    print('经训练后神经网络的参数w1:\n', sess.run(w1))
    print('经训练后神经网络的参数w2:\n', sess.run(w2))

