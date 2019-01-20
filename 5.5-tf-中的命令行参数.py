#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-20 21:39:53
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf


# 第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.flags.DEFINE_string('string_name', 'dana', "descrip1")
tf.flags.DEFINE_integer('init_name', 10, "descrip2")
tf.flags.DEFINE_boolean("bool_name", False, "descrip3")

FLAGS = tf.flags.FLAGS


def main(_):  # 必须带参数，否则：'TypeError: main() takes no arguments (1 given)';   main的参数名随意定义，无要求
    print("FLAGS.string_name:", FLAGS.string_name)
    print("FLAGS.init_name:", FLAGS.init_name)
    print("FLAGS.bool_name:", FLAGS.bool_name)


if __name__ == '__main__':
    tf.app.run()

# CMD中输入以下命名行参数
"""
python 5.5-tf-中的命令行参数.py
python 5.5-tf-中的命令行参数.py --string_name test_dir --init_name 99 --bool_name True
"""
