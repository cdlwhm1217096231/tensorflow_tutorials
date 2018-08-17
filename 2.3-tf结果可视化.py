#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-17 16:52:54
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
通常神经层都包括输入层、隐藏层和输出层。这里的输入层只有一个属性， 所以我们就只有一个输入
输入层1个节点、隐藏层10个节点、输出层1个节点的神经网络
输入层的维度是[n,1]
隐层的维度是  [1,10]
输出层的维度是[10,1]

权值矩阵的维度是：
weight1=[1,10]
bais1=[10,1]
weight2=[10,1]
bais2=[1,1]
'''


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# # 构造一个数据集
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


# 利用占位符定义我们所需的神经网络的输入
'''
 tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
'''
# placeholder 占个位
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# add output layer
# 上一层的输出是这一层的输入
prediction = add_layer(l1, 10, 1, activation_function=None)

# loss函数和使用梯度下降的方式来求解
loss = tf.reduce_mean(tf.reduce_sum(
    tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    for i in range(1000):
        # 在带有placeholder的变量里面，每一次sess.run 都需要给一个feed_dict，这个不能省略啊！
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
