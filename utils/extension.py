#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: extension.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import sys
import numpy as np
import os
from abc import abstractmethod

class Extension(object):
    def init(self):
        pass

    @abstractmethod
    def trigger(self):
        pass

class PeriodicExtension(Extension):
    def __init__(self, period):
        self.__period = period
        self.epoch_num = 0

    def init(self):
        pass

    def trigger(self):
        self.epoch_num += 1
        if self.epoch_num % self.__period == 0:
            self._trigger()

    @abstractmethod
    def _trigger(self):
        pass

class OnehotClassificationValidation(PeriodicExtension):
    """
        use with output: bxn probability
        and label: (b,) vector
    """
    def __init__(self, ds, prefix,
                 period=1,
                 input_op_name='input',
                 label_op_name='label',
                 output_op_name='output'):
        super(OnehotClassificationValidation, self).__init__(period)
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
            self.nr_correct_var = tf.reduce_sum(tf.cast(correct, tf.int32))

    def _trigger(self):
        cnt = 0
        cnt_correct = 0
        for (img, label) in self.ds.get_data():
            # TODO dropout?
            feed = {self.input_var: img, self.label_var: label}
            cnt += img.shape[0]
            cnt_correct += self.nr_correct_var.eval(feed_dict=feed)
        # TODO write to summary?
        print "Accuracy at epoch {}: {}".format(
            self.epoch_num, cnt_correct * 1.0 / cnt)


class PeriodicSaver(PeriodicExtension):
    def __init__(self, log_dir, period=1):
        super(PeriodicSaver, self).__init__(period)
        self.path = os.path.join(log_dir, 'model')

    def init(self):
        self.saver = tf.train.Saver(max_to_keep=99999)

    def _trigger(self):
        self.saver.save(tf.get_default_session(), self.path,
                        global_step=self.epoch_num, latest_filename='latest')
