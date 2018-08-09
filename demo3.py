#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.3.3

__date__ = '2018/7/30 14:24'
__author__ = 'cdl'

######################通过placeholder实现前向传播算法###################
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 定义placeholder作为存放输入数据的地方，这里维度也不一定要定义；但如果维度是确定的，那么给出维度可以降低出错的概率
x = tf.placeholder(tf.float32, shape=(3, 2), name='input')  # 不使用常量来表示选取的数据，将会使得计算图将会太大
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # print(sess(y))
    print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
#####################以下代码定义了一个简单的损失函数####################

# 使用sigmoid函数将y转换成0到1之间的数值，转化后y代表预测正样本的概率，1-y代表预测负样本的概率
y = tf.sigmoid(y)
# 定义损失函数（交叉熵）来刻画预测值与实际值之间的差距
cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y)* tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
# 定义学习率
learning_rate = 0.001
# 定义反向传播的优化方法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
sess.run(train_step)
