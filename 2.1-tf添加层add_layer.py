#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-17 15:55:10
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf


'''
在 Tensorflow 里定义一个添加层的函数可以很容易的添加神经层,为之后的添加省下不少时间
神经层里常见的参数通常有weights、biases和激励函数
定义添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数
我们设定默认的激励函数是None

inputs =[样本数 * 特征数] ，而吴恩达的教程是 特征数*样本数
所以，这里的表示方式是： input * weights
假如，输入层的结点个数是2，隐层是3
inputs=[n*2]  ,weihts=[2*3] ,bias=[1,3]
inputs*weigths=[n,3] + bias=[1,3] ，这样的矩阵维度相加的时候，python会执行它的广播机制
so,这一层的输出的维度是 [n,3]
'''


def add_layer(inputs, in_size, out_size, activation_function=None):
        # 因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_biases = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_biases
    else:
        outputs = activation_function(Wx_plus_biases)
    return outputs
