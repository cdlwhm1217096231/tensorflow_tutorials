
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-08 22:32:55
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

"""不使用正则化"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 1.设置输入和输出节点的个数,配置神经网络的参数
INPUT_NODE = 784  # 输入节点
OUT_NODE = 10      # 输出节点
LAYER1_NODE = 500  # 隐藏层节点数

BATCH_SIZE = 100  # 每个batch打包的样本个数

# 模型相关的参数
LEARING_RATE_BASE = 0.8
LEARING_RETE_DECAY = 0.99

TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 2.定义辅助函数来计算前向传播结果，使用ReLU做为激活函数


def interfence(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
        return tf.matmul(layer1, weight2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(
            weight1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)
# 3.定义训练过程


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_actual = tf.placeholder(tf.float32, [None, OUT_NODE], name='y-input')
    # 生成隐藏层的参数
    weight1 = tf.Variable(tf.truncated_normal(
        [INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weight2 = tf.Variable(tf.truncated_normal(
        [LAYER1_NODE, OUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUT_NODE]))
    # 计算不含滑动平均模型的前向传播的结果
    y = interfence(x, None, weight1, biases1, weight2, biases2)

    # 定义训练轮数和相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = interfence(x, variable_averages, weight1,
                           biases1, weight2, biases2)
    # 计算交叉熵及平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_actual, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 损失函数计算
    loss = cross_entropy_mean
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARING_RETE_DECAY, staircase=True)
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)
    # 反向传播更新参数和更新每个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    # 计算正确率
    correct_prediction = tf.equal(
        tf.argmax(average_y, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # 初始化会话，并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images,
                         y_actual: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_actual: mnist.test.labels}
  # 循环的训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('经过%s训练，模型在验证集上的准确率为:%s' % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_actual: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('经过%s训练，模型在测试集上的准确率为:%s' % (i, test_acc))


def main(argv=None):
    # 载入数据集
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
