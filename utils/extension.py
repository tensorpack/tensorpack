#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: extension.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np

class OnehotClassificationValidation(object):
    """
        use with output: bxn probability
        and label: (b,) vector
    """
    def __init__(self, ds, prefix,
                 input_op_name='input',
                 label_op_name='label',
                 output_op_name='output'):
        self.ds = ds
        self.input_op_name = input_op_name
        self.output_op_name = output_op_name
        self.label_op_name = label_op_name

    def init(self):
        self.graph = tf.get_default_graph()
        with tf.name_scope('validation'):
            self.input_var = self.graph.get_operation_by_name(self.input_op_name).outputs[0]
            self.label_var = self.graph.get_operation_by_name(self.label_op_name).outputs[0]
            self.output_var = self.graph.get_operation_by_name(self.output_op_name).outputs[0]

            correct = tf.equal(tf.cast(tf.argmax(self.output_var, 1), tf.int32),
                               self.label_var)
            # TODO: add cost
            self.accuracy_var = tf.reduce_mean(tf.cast(correct, tf.float32))

    def trigger(self):
        scores = []
        for (img, label) in self.ds.get_data():
            feed = {self.input_var: img, self.label_var: label}
            scores.append(
                self.accuracy_var.eval(feed_dict=feed))
        acc = np.array(scores, dtype='float32').mean()
        # TODO write to summary?
        print "Accuracy: ", acc

