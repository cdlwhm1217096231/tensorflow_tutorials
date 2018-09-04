#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-04 16:22:16
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf


"""深层循环神经网络的实现"""


lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
# 通过MultiRNNCell类实现深层循环神经网络中每个时刻的前向传播过程，number_of_layers表示有多少层
# 初始化MultiRNNCell，否则每层之间会共享参数
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_size) for _ in range(number_of_layers)])
# 通过zero_state函数来获取初始状态
state = stacked_lstm.zero_state(batch_size, tf.float32)
# 计算每一时刻的前向传播结果
for i in range(len(num_steps)):
    if i>0:
        tf.get_variable_scope().reuse_variables()
    stacked_lstm_output, state = stacked_lstm(current_input, state)
    final_output = fully_connected(stacked_lstm_output)
    loss += calc_loss(final_output, expected_output)

