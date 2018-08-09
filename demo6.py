#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.3.3

__date__ = '2018/8/1 9:35'
__author__ = 'cdl'

import tensorflow as tf
import numpy as np

# 使用正则化处理过拟合问题
weight = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
    print('L1正则化结果，lambda=0.5时:', sess.run(tf.contrib.layers.l1_regularizer(.5)(weight)))  # L1正则
    print('L2正则化结果，lambda=0.5时: %s' % sess.run(tf.contrib.layers.l2_regularizer(.5)(weight)))  # L2正则
