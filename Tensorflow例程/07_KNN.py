#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-12 09:51:41
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./dataset/MNIST_data/", one_hot=True)

x_train, y_train = mnist.train.next_batch(5000)   # 5000个训练样本
x_test, y_test = mnist.test.next_batch(200)  # 200个测试样本

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [784])

# 距离度量:使用L1 norm
distance = tf.reduce_sum(tf.abs(tf.add(X, tf.negative(Y))),
                         axis=1)  # tf.negative是对一个数取反
# distance返回的是N个训练样本与单个测试样本的距离之和
y_hat = tf.argmin(distance, axis=0)  # 返回最小距离所对应的索引
accuracy = 0.0
init = tf.global_variables_initializer()
# 启动会话
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(x_test)):
        # 获取最近的样本点所对应的索引
        nn_index = sess.run(y_hat, feed_dict={X: x_train, Y: x_test[i, :]})
        # 获取最近的样本点类别标签与真实的标签进行对比
        print("Test Sample", i + 1, "Prediction:", np.argmax(
            y_train[nn_index]), "True Class:", np.argmax(y_test[i]))
        # 计算精度
        if np.argmax(y_train[nn_index]) == np.argmax(y_test[i]):
            accuracy += 1.0 / len(x_test)
    print("Done!")
    print("Accuracy: %.2f%%" % (accuracy * 100))
