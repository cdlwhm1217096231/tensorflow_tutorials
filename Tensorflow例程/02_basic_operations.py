#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-11 08:13:43
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf

# 基本的常量运算
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a: %i" % sess.run(a), "b: %i" % sess.run(b))
    print("a+b: %i" % sess.run(a + b))
    print("a*b: %i" % sess.run(a * b))

# 使用变量作为图的输入
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
# 启动默认的图
with tf.Session() as sess:
    # 每次操作都对变量进行
    print("变量的加法: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("变量的乘法: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

# 矩阵的乘法
m1 = tf.constant([[3., 3.]])
m2 = tf.constant([[2.], [2.]])
product = tf.matmul(m1, m2)
with tf.Session() as sess:
	result = sess.run(product)
	print(result)
