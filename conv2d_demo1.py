#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-15 17:20:06
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf
import numpy as np

"""卷积层"""
# 1.输入矩阵

M = np.array([
    [[1], [-1], [0]],
    [[-1], [2], [1]],
    [[0], [2], [-2]]
])
print('输入矩阵的形状:', M.shape)
# 2.定义卷积过滤器,深度为1
filter_weight = tf.get_variable(
    'weight', [2, 2, 1, 1], initializer=tf.constant_initializer([[1, -1], [0, 2]]))
biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(1))
# 3. 调整输入的格式符合TensorFlow的要求
M = np.asarray(M, dtype='float32')
M = M.reshape(1, 3, 3, 1)
# 4. 计算矩阵通过卷积层过滤器和池化层过滤器计算后的结果
x = tf.placeholder('float32', [1, None, None, 1])
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 2, 2, 1], padding='SAME')
bias = tf.nn.bias_add(conv, biases)
pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[
                      1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convoluted_M = sess.run(bias, feed_dict={x: M})
    pooled_M = sess.run(pool, feed_dict={x: M})
    print('卷积层之后的结果:\n', convoluted_M)
    print('池化层之后的结果:\n', pooled_M)
