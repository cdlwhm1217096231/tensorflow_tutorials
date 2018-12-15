#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-11 08:30:43
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from __future__ import print_function, division, print_function
import tensorflow as tf
import numpy as np
"""
tensorflow通过图将计算的定义与执行分开，提供一种声明式的编程模型，但是debug不方便，不够pytho
无法使用python原生的控制语句与数据结构，因此产生了eager Execution
"""

print("设置Eager模式...")
tf.enable_eager_execution()
tfe = tf.contrib.eager
print("定义tensor常量...")
a = tf.constant(2)
print("a = %d" % a)
b = tf.constant(3)
print("b = %d" % b)
# 运行操作，不需要使用tf.Session()
c = a + b
print("a+b = %i" % c)
d = a * b
print("a*b = %i" % d)
# 完全与numpy兼容

# 定义常量
a = tf.constant([[2., 1.], [1., 0.]], dtype=tf.float32)
print("张量a:\n a=%s" % a)
b = np.array([[3., 0.], [5., 1.]], dtype=np.float32)
print("numpy Array b:\n b=%s" % b)
# 不需要tf.Session，开始运算
c = a + b
print("a+b = %s" % c)
d = tf.matmul(a, b)
print("a*b = %s" % d)

# 遍历tensor a
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])
