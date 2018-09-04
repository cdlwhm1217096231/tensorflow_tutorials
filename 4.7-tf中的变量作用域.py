#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-03 14:56:21
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf

# variable_scope()示例
"""
tensorflow中通过变量名称获取变量的机制主要是通过tf.get_variable和tf.variable_scope函数实现的
tf提供tf.get_variable函数来创建或获取变量；当tf.get_variable用于创建变量时，它和tf.Variable的功能基本等价
"""
# tf.get_variable函数调用时提供维度信息和初始化方法，tf.get_variable函数与tf.Variable函数最大区别是：tf.Variable函数中变量名称是一个可选参数
# tf.get_variable函数中的变量名称是一个必填参数，tf.get_variable会根据这个名字去创建或获取一个变量
v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0, shape=[1], name="v"))
# 为了解决出现变量复用造成的tf.get_variable错误，需要通过tf.variable_scope()函数生成一个上下文管理器，并明确指出在这个上下文管理器中，
# tf.get_variable()将获取已经生成的变量
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
"""
由于命名空间foo中已经存在名为v的变量，所以下面代码将会报错
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
"""
# 在生成上下文管理器时，将参数reuse设置为True,这样tf.get_variable函数将直接获取已经声明的变量
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v1 == v)  # v,v1代表的是相同的tf中的变量
# 将reuse设置为True时，tf.variable_scope将只能获取已经创建过的变量,因为在bar命名空间中，还没有创建过变量v，所以报错
"""
with tf.variable_scope("bar", reuse=True):
    v = tf.get_variable("v",[1])
"""
# 如果tf.variable_scope()函数使用参数reuse=None,reuse=False创建上下文管理器时，tf.get_variable操作将会创建新的变量,如果同名变量已存在，将会报错
# tf.variable_scope()函数是可以嵌套的
with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)
    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)
    print(tf.get_variable_scope().reuse)
# tf.variable_scope()函数除了可以控制tf.get_variable执行功能之外，也提供了一个管理变量命名空间的方式
with tf.variable_scope("hi"):
    v1 = tf.get_variable("v", [1])
    print(v1.name)
    print(tf.get_variable_scope().reuse)

with tf.variable_scope("hi"):
    with tf.variable_scope("bar"):
        v2 = tf.get_variable("v", [1])
        print(v2.name)
    v4 = tf.get_variable("v1", [1])  # 此处的变量名一定不能是v
    print(v4.name)

with tf.variable_scope("", reuse=True):
    v5 = tf.get_variable("hi/bar/v", [1])
    print(v5 == v2)
    v6 = tf.get_variable("hi/v1", [1])
    print(v6 == v4)

# name_scope()在可视化过程中，为变量划分范围，表示计算图中的一个层级,不影响get_variable创建的变量，只会影响Variable()创建的变量
"""
name_scope 对用get_variable()创建的变量的名字不会有任何影响，而 Variable()创建的操作会被加上前缀，并且会给操作加上名字前缀
"""
with tf.variable_scope("fooo"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        b = tf.Variable(tf.zeros([1]), name="b")
        x = 1.0 + v
        print(v.name)
        print(b.name)
        print(x.op.name)
