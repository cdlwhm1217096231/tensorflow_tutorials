#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-16 21:12:38
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
product = tf.matmul(matrix1, matrix2)  # 矩阵乘法
# 会话控制方式1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
# 会话控制方式2
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
