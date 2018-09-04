#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-04 14:48:57
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf

# 定义一个LSTM结构

# 定义一个lstm结构，在tensorflow中通过一句话就能实现一个完整的lstm结构
# lstm中使用的变量也会在该函数中自动被声明
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

# 将lstm中的状态初始化为全0数组，BasicLSTMCell提供了zero_state来生成全0数组
# 在优化RNN时每次也会使用一个batch的训练样本，batch_size给出了一个batch的大小
state = lstm.zero_state(batch_size, tf.float32)

# 定义损失函数
loss = 0.0
# 为了在训练中避免梯度弥散的情况，规定一个最大的序列长度num_steps
for i in range(num_steps):
    # 在第一个时刻声明lstm结构中使用的变量，在之后的时刻都需要重复使用之前定义好的变量
    if i>0:
        tf.get_variable_scope().reuse_variables()
    # 每一步处理时间序列中的一个时刻，将当前输入current_inputa(xt)和前一时刻状态state(ht-1和ct-1)传入LSTM结构
    # 就可以得到当前lstm结构的输出lstm_output(ht)和更新后的状态state(ct)
    lstm_output, state = lstm(current_input, state)
    # 将当前时刻lstm输出传入一个全连接层得到最后的输出
    final_output = fully_connected(lstm_output)
    # 计算当前时刻输出的损失
    loss += calc_loss(final_output, expected_output)

