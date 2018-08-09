#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.3.3

__date__ = '2018/7/31 8:50'
__author__ = 'cdl'

import tensorflow as tf

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
c = tf.constant([1.0, 2.0, 3.0])
v1 = tf.constant([[1.0, 2.0], [3.0, 5.0]])
v2 = tf.constant([[3.0, 2.0], [6.0, 7.0]])
with tf.Session() as sess:
    print(tf.clip_by_value(v, 2.5, 4.5).eval())
    print(tf.log(c).eval())
    print((v1*v2).eval())  # 两个矩阵对应元素之间的相乘
    print(tf.matmul(v1, v2).eval())  # 矩阵乘法
    print(tf.reduce_mean(v).eval())


