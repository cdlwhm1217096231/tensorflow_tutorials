#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-12 08:34:16
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from __future__ import print_function, absolute_import, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# set eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager

Traing_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
            7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
Traing_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
            2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
n_samples = len(Traing_X)

# 超参数设置
learning_rate = 0.01
display_step = 100
num_steps = 1000

# Weights and bias
W = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())


# Linear regression (Wx+b)
def linear_regression(inputs):
    return inputs * W + b


# Mean square error
def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2)) / (2 * n_samples)


# SGD optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# compute gradient
gradient = tfe.implicit_gradients(mean_square_fn)
# Initial cost, before optimizing
print("Initial cost={:.4f}".format(mean_square_fn(
    linear_regression, Traing_X, Traing_Y)), "W=", W.numpy(), "b=", b.numpy())
# Training
for step in range(num_steps):
    optimizer.apply_gradients(gradient(linear_regression, Traing_X, Traing_Y))
    if (step + 1) % display_step == 0 or step == 0:
        print("Epoch: %04d" % (step + 1), "cost={:.9f}".format(mean_square_fn(
            linear_regression, Traing_X, Traing_Y)), "W=", W.numpy(), "b=", b.numpy())

# plot graph
plt.plot(Traing_X, Traing_Y, "ro", label="Original data")
plt.plot(Traing_X, np.array(W * Traing_X + b), label="Fitted line")
plt.legend()
plt.show()
