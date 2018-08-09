#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.3.3

__date__ = '2018/8/1 11:08'
__author__ = 'cdl'

import tensorflow as tf
import numpy as np

############### 通过集合计算一个5层神经网络带L2正则的损失函数方法################

# 获取上一层神经网络的权重，并将这个权重的L2正则化损失加入变量losses的集合中


def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # add_to_collection函数将整个新生成变量的L2正则化损失项加入集合，这个函数的第一个参数losses是集合的名字，第二个参数是要加入这个集合的内容
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# 定义每层网络中的节点个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)
# 这个变量维护前向传播时最深层的节点，开始的时候是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]
# 通过一个循环生成5层全连接的神经网络结构
for i in range(1, n_layers):
    # layer_dimension[i]为下一层的节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用relu激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点个数更新为当前层的节点个数
    in_dimension = layer_dimension[i]
# 损失函数，L2正则项已经加入计算图上的集合
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方误差损失函数加入损失集合中
tf.add_to_collection('losses', mse_loss)
# get_collection返回一个列表，这个列表是所有这个集合中的元素。例中这些元素是损失函数的不同部分，将它们加起来就是最终的损失函数
loss = tf.add_n(tf.get_collection('losses'))
print(loss)
