#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-15 18:32:51
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf

"""实现卷积层的前向传播过程"""

x = tf.placeholder('float32', [1, None, None, 1])
"""通过get_variable()方式创建过滤器的权重变量和偏置项变量
卷积层的参数只和过滤器的尺寸、过滤器的深度、当前层节点矩阵的深度有关，前两个参数代表过滤器的尺寸、第三个参数代表当前层的深度、第四个参数代表过滤器的深度"""
filter_weight = tf.get_variable(
    'weights', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
# 偏置项参数个数等于过滤器的深度
biases = tf.get_variable(
    'biases', [16], initializer=tf.constant_initializer(0.1))
# tf.nn.conv2d()函数提供了一个非常方便的函数实现卷积层的前向传播算法，第一个参数是当前层的节点矩阵，（注意这个节点矩阵是一个四维矩阵，后面三维对应一个节点矩阵，第一维对应一个输入的batch，例如:input[0,:,:,:]表示第一张图片）；第二个参数提供了卷积层的权重，第三个参数提供不同维的步长，最后一个参数是填充，tensorflow提供两种选择SAME和VALID，SAME表示全0填充
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 1, 1, 1], padding='SAME')
# tf.nn.bais_add提供一个给每个节点加上偏置项的函数
bias = tf.nn.bias_add(conv, biases)
# 将计算结果通过RELU激活去线性化
actived_conv = tf.nn.relu(bias)
