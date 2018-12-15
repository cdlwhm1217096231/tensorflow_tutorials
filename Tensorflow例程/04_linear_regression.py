#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-12 08:09:08
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
rng = np.random


# 超参数设置
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# 训练数据
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]
# tf graph input
X = tf.placeholder("float")
Y = tf.placeholder("float")
# set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
# construct a linear model
y_hat = tf.add(tf.multiply(X, W), b)
# Mean squared error
cost = tf.reduce_sum(tf.pow(y_hat - Y, 2)) / (2 * n_samples)
# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initialize the variables
init = tf.global_variables_initializer()
# start session
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # display logs perepoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch: %04d" % (epoch + 1),
                  "cost={:.4f}".format(c), "W =", sess.run(W), "b =", sess.run(b))
    print("优化完成!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training_cost: ", training_cost, "W = ",
          sess.run(W), "b = ", sess.run(b))

    # plot graph
    plt.plot(train_X, train_Y, "ro", label="Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()
