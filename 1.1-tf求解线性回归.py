#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-16 20:57:15
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf
import numpy as np

# create data
# 创建一个随机的数据集
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure start
# 随机初始化 权重
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# 估计的y值
y = Weights * x_data + biases
# 估计的y和真实的y，计算cost
loss = tf.reduce_mean(tf.square(y - y_data))
# 梯度下降优化
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
"""
到目前为止, 我们只是建立了神经网络的结构, 还没有使用这个结构.
在使用这个结构之前, 我们必须先初始化所有之前定义的Variable, 所以这一步是很重要的
"""
# 训练
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)    #  用 Session来 run 每一次 training 的数据
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print('经过%s次训练后,权重%s,偏置项%s' %
                  (step, sess.run(Weights), sess.run(biases)))
