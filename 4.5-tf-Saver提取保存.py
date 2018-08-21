#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2

import tensorflow as tf
import numpy as np
# remeber to define the same dtype and shape when restore

# # 保存
# W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "my_net/save_net.ckpt")
#     print("Saver to path:", save_path)

# 恢复
# remeber to redefine the same dtype and shape for your variables
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

# not need initial step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print('weights:', sess.run(W))
    print('biases:', sess.run(b))