# !/usr/bin/env python2
#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf


def one_hot(y, num_labels):
    batch_size = y.get_shape().as_list()[0]
    assert type(batch_size) == int, type(batch_size)
    y = tf.expand_dims(y, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, y])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, num_labels]), 1.0, 0.0)
    onehot_labels.set_shape([batch_size, num_labels])
    return tf.cast(onehot_labels, tf.float32)
