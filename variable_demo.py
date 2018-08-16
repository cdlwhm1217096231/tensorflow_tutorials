#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-14 14:33:00
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf

"""创建变量"""
v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0, shape=[1]), name="v")

with tf.variable_scope("foo"):
    v = tf.get_variable(
        "v", shape=[1], initializer=tf.constant_initializer(1.0))
    # v = tf.get_variable("v", [1])
# 注意两者区别
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v == v1)


# reuse=True,tf.variable_scope将只能获取已经创建过的变量
with tf.variable_scope("bar"):
    v = tf.get_variable(
        "v", shape=[1], initializer=tf.constant_initializer(1.0))
with tf.variable_scope("bar", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v1 == v)
