#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-11 08:08:26
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$


import tensorflow as tf
print(tf.__version__)


hello = tf.constant("hello world!")
# 开始会话
with tf.Session() as sess:
    # 运行图
    print(sess.run(hello))
