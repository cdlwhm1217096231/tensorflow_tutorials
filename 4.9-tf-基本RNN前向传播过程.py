#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-04 14:53:25
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np

state0 = [0, 0]
x_input = [1, 2]

cell_w = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
cell_bias = np.array([0.1, -0.1])
print(cell_w.shape)
print(cell_bias.shape)
state0.append(x_input[0])
state0 = np.array([state0])
output_w = np.array([1.0, 2.0])
output_2d_w = output_w[:, np.newaxis]
# print(output_2d_w)
output_bias = 0.1
# 第一个细胞体中的环节
before_activation = np.dot(state0, cell_w) + cell_bias
print('before_activation:', before_activation)
state1 = np.tanh(before_activation)
print("下一个时刻的输出状态state1:", state1)
final_output = np.dot(state1, output_2d_w) + output_bias
print("final_output:", final_output)
print('--------------分隔线-----------')
# 第二个细胞体中的环节
state2 = np.append(state1, x_input[1])
# print("state1:", state2)
state2 = state2[np.newaxis, :]
before_activation1 = np.dot(state2, cell_w) + cell_bias
print("before_activation1:", before_activation1)
state3 = np.tanh(before_activation1)
print("下一个时刻的输出状态state3:", state3)
final_output1 = np.dot(state3, output_2d_w) + output_bias
print("final_output1:", final_output1)