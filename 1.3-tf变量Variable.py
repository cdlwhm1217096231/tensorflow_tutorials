#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-16 21:19:33
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$
import tensorflow as tf

# 定义一个变量
state = tf.Variable(0, name='counter')
# print(state.name)

# 定义一个常量
one = tf.constant(1)
# 定义一个加法
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

#  初始化，在初始化之前是变量是没有值的
init = tf.global_variables_initializer()  # must have if define variable
with tf.Session() as sess:
	# 这里变量还是没有被激活，需要再在 sess 里, sess.run(init) , 激活 init 这一步
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
