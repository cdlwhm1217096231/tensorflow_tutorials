#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-12 09:28:49
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from __future__ import absolute_import, division, print_function
import tensorflow as tf

# set eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./dataset/MNIST_data/", one_hot=False)
# 超参数设置
learning_rate = 0.1
batch_size = 128
num_step = 1000
display_step = 100
# iterator for the dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels))
dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
dataset_iter = tfe.Iterator(dataset)
# Variable
W = tfe.Variable(tf.zeros([784, 10]), name="Weights")
b = tfe.Variable(tf.zeros([10]), name="bias")


# logistic regression(Wx+b)
def logistic_regression(inputs):
    return tf.matmul(inputs, W) + b


# cross_entropy loss function
def loss_fn(inference_fn, inputs, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=inference_fn(inputs), labels=labels))


# calculate accuracy
def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# SGD optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# compute gradient
gradient = tfe.implicit_gradients(loss_fn)
# Training model
average_loss = 0.
average_acc = 0.
for step in range(num_step):
    # iterate through the dataset
    d = dataset_iter.next()
    # Images
    x_batch = d[0]
    y_batch = tf.cast(d[1], dtype=tf.int64)
    # compute the batch loss
    batch_loss = loss_fn(logistic_regression, x_batch, y_batch)
    batch_acc = accuracy_fn(logistic_regression, x_batch, y_batch)
    average_loss += batch_loss
    average_acc += batch_acc
    if step == 0:
        # display the initial cost, before optimizing
        print("Initial loss={:.4f}".format(average_loss))
    # Update the variables following gradients info
    optimizer.apply_gradients(gradient(logistic_regression, x_batch, y_batch))

    # Display info
    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print("Step:", '%04d' % (step + 1), " loss=",
              "{:.9f}".format(average_loss), " accuracy=",
              "{:.4f}".format(average_acc))
        average_loss = 0.
        average_acc = 0.


# Evaluate model on the test image set
testX = mnist.test.images
testY = mnist.test.labels
test_acc = accuracy_fn(logistic_regression, testX, testY)
print("Testset Accuracy: {:.4f}".format(test_acc))
