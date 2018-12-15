#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-12 08:53:36
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据集
mnist = input_data.read_data_sets("./dataset/MNIST_data/", one_hot=True)
# 超参数设置
learning_rate = 0.01
training_epochs = 1000
batch_size = 100
display_step = 50
# tf graph input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])    # 10个类别
# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]) + 0.1)
# construct model
y_hat = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax
# minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=1))
# gradient descent
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cost)
# Initialize the variables
init = tf.global_variables_initializer()
# start session
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples /
                          batch_size)   # 总共有多少个batch
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Fitting training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={
                            x: batch_x, y: batch_y})
            # compute average loss
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            print("Epoch: %04d" % (epoch + 1), "cost={:.4f}".format(avg_cost))
    print("---------------优化完成!---------------")
    # Test model
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    # calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval(
        {x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
