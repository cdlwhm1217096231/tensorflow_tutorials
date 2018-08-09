#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.3.3

__date__ = '2018/7/30 10:35'
__author__ = 'cdl'

import tensorflow as tf


weights = tf.Variable(tf.random_normal([2, 3], stddev=2))  # 变量声明函数，并给其进行随机初始化，生成2*3的矩阵，矩阵中的元素全是0，且标准差为2
zeros = tf.zeros([2, 4])  # 产生全0的数组
ones = tf.ones([2, 3])   # 产生全1的数组
constant = tf.constant([1, 2])  # 产生一个给定值的常量


###################通过变量实现神经网络的参数并实现前向传播的过程#####################
import tensorflow as tf

# 声明w1、w2两个变量，还通过seed参数设定了随机种子，这样可以保证每次运行得到的结果是一样的。
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 暂时将输入的特征向量定义为一个常量, x是1*2的矩阵
x = tf.constant([[0.7, 0.9]])
# 前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 初始化w1、w2两个变量
"""
sess = tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(y))
sess.close()   #  此句不能少，防止内存泄漏
"""
# 推荐写法
with tf.Session() as sess:
    sess.run(w1.initializer)  # 变量的初始化，变量声明时只是给出了变量初始化的方法，但这个方法并没有真正运行
    sess.run(w2.initializer)
    print(sess.run(y))
# tf提供了一种更加便捷的方法完成变量的初始化，通过函数tf.global_variables_initializer函数实现初始化所有变量的过程
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
sess.close()




