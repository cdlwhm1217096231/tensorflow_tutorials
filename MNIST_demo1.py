#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""整个模型"""

# MNIST数据集相关参数
INPUT_NODE = 784  # 输入层神经元个数
OUTPUT_NODE = 10  # 输出层神经元个数

# 配置神经网络参数
LAYER1_NODE = 500  # 隐藏层神经元个数
BATCH_SIZE = 100  # 一个训练batch中的训练样本数。数据越小，越接近随机梯度下降；数据越大，越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARAZTION_RATE = 0.0001  # 描述模型复杂度的正则项在损失函数中的系数lambda
TRAINING_STEPS = 30000  # 训练次数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均初始的衰减率

#  定义辅助函数来计算前向传播结果，使用ReLU做为激活函数,给定神经网络的输入和所有参数，计算神经网络的前向传播结果，支持传入计算参数平均值的类，方便使用滑动平均模型


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类variable_averages传入avg_class这个形参，通过avg.average()获取滑动平均之后变量的取值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(
            weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型的过程


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal(
        [INPUT_NODE, LAYER1_NODE], stddev=0.1))  #  tf.truncated_normal从截断的正态分布中输出随机值，stddev: 正态分布的标准差，tf.random_normal从正态分布中输出随机值，stddev: 正态分布的标准差， seed: 一个整数，当设置之后，每次生成的随机数都一样
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal(
        [LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数
    global_step = tf.Variable(0, trainable=False)  # 控制滑动衰减率的变量global_step

    # 定义滑动平均类variable_averages
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())  # 定义一个更新变量的滑动平均操作
    average_y = inference(x, variable_averages, weights1,
                          biases1, weights2, biases2)  # 计算含滑动平均类的前向传播结果

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)   # 计算l2正则项
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion  #   带L2正则化的损失函数
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()  # 初始化所有变量
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("经过%d次训练, 在验证集上的准确率为: %s " % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("经过%d次训练, 在测试集上神经网络模型的准确率为: %s" % (TRAINING_STEPS, test_acc)))


def main(argv=None):
    # 载入数据集
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
