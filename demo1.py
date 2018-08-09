#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.3.3

__date__ = '2018/7/30 9:51'
__author__ = 'cdl'

import tensorflow as tf

input1 = tf.constant([1.0, 2.0, 3.0], name='input1')  # 定义两个常量tf.constant
input2 = tf.constant([3.0, 2.0, 1.0], name='input2')
output = tf.add_n([input1, input2], name='output')
print(output)  # 张量中保存了三个属性: 名字、维度、类型
init = tf.global_variables_initializer()  # 初始化
merged = tf.summary.merge_all()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('log_test', sess.graph)
    sess.run(init)
    print((sess.run(output)))
writer.close()

