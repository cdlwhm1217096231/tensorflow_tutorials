#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.3.3

__date__ = '2018/8/6 9:02'
__author__ = 'cdl'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
###### 接下来就是进行“三步走”的实践过程,选择最简单的网络结构，依旧选择二次代价函数和梯度下降算法进行优化#####
print('训练样本信息:', mnist.train.images.shape, mnist.train.labels.shape)
print('测试样本信息:', mnist.test.images.shape, mnist.test.labels.shape)
print('验证样本信息:', mnist.validation.images.shape, mnist.validation.labels.shape)
# 每次从训练集中所取的数据的大小
batch_size = 100
# 计算一共有多少个batch
n_batch = mnist.train.num_examples
# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y_actual = tf.placeholder(tf.float32, [None, 10])
# 创建一个简单的神经网络
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_prediction = tf.nn.softmax(tf.matmul(x, w) + b )
# 交叉熵代价函数
loss = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_prediction), reduction_indices=[1]))
# 使用梯度下降法进行优化
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#######在会话中执行#########

# 初始化变量
init = tf.global_variables_initializer()
# 结果存放在一个布尔型的列表中
# argmax返回一维张量中最大值的索引
correct_predicition = tf.equal(tf.argmax(y_actual, 1), tf.argmax(y_prediction, 1))  # 此处获得是一个布尔型的列表
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})
        acc = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})
        print('经迭代%s次后,测试准确率为:%s' % (epoch, acc))
