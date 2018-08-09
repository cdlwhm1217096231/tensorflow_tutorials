#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.3.3

__date__ = '2018/8/1 8:45'
__author__ = 'cdl'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 直线的拟合

# 创建100个随机点
x_data = np.random.rand(20)
y_data = x_data * 5 + 1
# 构造线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b
# 定义损失函数
loss = tf.reduce_mean(tf.square(y_data - y))
# 选用梯度下降优化算法
optimizer = tf.train.GradientDescentOptimizer(0.3)
# 最小化代价函数
train = optimizer.minimize(loss)
# 初始化全部变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(101):
        sess.run(train)
        if step % 10 == 0:
            print('第%s次的训练结果是:%s' % (step, sess.run([k, b])))
# 线性回归

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])
# 神经网络中间层(10个神经元）
weight_L1 = tf.Variable(tf.random_normal([1, 10]))
baises_L1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_b_L1 = tf.matmul(x, weight_L1) + baises_L1
#  激活函数tanh
l1 = tf.nn.tanh(wx_plus_b_L1)
# 神经网络输出层
weight_L2 = tf.Variable(tf.random_normal([10, 1]))
baises_L2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_L2 = tf.matmul(l1, weight_L2) + baises_L2
#  激活函数tanh
y_prediction = tf.nn.tanh(wx_plus_b_L2)
# 损失函数----二次代价函数
loss = tf.reduce_mean(tf.square(y - y_prediction))
# 使用梯度下降算法进行优化
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    # 初始化全部变量
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(2001):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 获得预测值
    prediction_value = sess.run(y_prediction, feed_dict={x: x_data})
    # 绘图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=2.5)
    plt.title('神经网络训练进行线性拟合')
    plt.show()
