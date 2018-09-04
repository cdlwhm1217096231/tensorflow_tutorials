#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-04 16:33:37
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf


"""
注： RNN中也有 dropout 方法，但是RNN一般只在不同层循环体结构之间使用dropout，而不在同一层传递的时候使用。 
在tensorflow中，使用tf.nn.rnn_cell.DropoutWrapper类可以很容易实现dropout功能。
"""
# 使用DropoutWrapper类来实现dropout功能，可以通过两个参数来控制dropout概率
# input_keep_prob用来控制输入的dropout概率，output_keep_prob用来控制输出的dropout概率
# output_keep_prob=0.9为被保留的数据为90%，将10%的数据随机丢弃
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.9)
