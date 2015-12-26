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
                 input_var_name='input:0',
                 label_var_name='label:0',
                 output_var_name='output:0'):
        super(OnehotClassificationValidation, self).__init__(period)
        self.ds = ds
        self.input_var_name = input_var_name
        self.output_var_name = output_var_name
        self.label_var_name = label_var_name

    def init(self):
        self.graph = tf.get_default_graph()
        with tf.name_scope('validation'):
            self.input_var = self.graph.get_tensor_by_name(self.input_var_name)
            self.label_var = self.graph.get_tensor_by_name(self.label_var_name)
            self.output_var = self.graph.get_tensor_by_name(self.output_var_name)
            self.dropout_var = self.graph.get_tensor_by_name('dropout_prob:0')

            correct = tf.equal(tf.cast(tf.argmax(self.output_var, 1), tf.int32),
                               self.label_var)
            self.nr_correct_var = tf.reduce_sum(tf.cast(correct, tf.int32))
            self.cost_var = self.graph.get_tensor_by_name('cost:0')

    def _trigger(self):
        cnt = 0
        correct_stat = Accuracy()
        sess = tf.get_default_session()
        cost_sum = 0
        for (img, label) in self.ds.get_data():
            feed = {self.input_var: img,
                    self.label_var: label,
                    self.dropout_var: 1.0}
            cnt += img.shape[0]
            correct, cost = sess.run([self.nr_correct_var, self.cost_var],
                                    feed_dict=feed)
            correct_stat.feed(correct, cnt)
            cost_sum += cost * cnt
        cost_sum /= cnt
        # TODO write to summary?
        print "After epoch {}: acc={}, cost={}".format(
            self.epoch_num, correct_stat.accuracy, cost_sum)


class PeriodicSaver(PeriodicExtension):
    def __init__(self, log_dir, period=1):
        super(PeriodicSaver, self).__init__(period)
        self.path = os.path.join(log_dir, 'model')

    def init(self):
        self.saver = tf.train.Saver(max_to_keep=99999)

    def _trigger(self):
        self.saver.save(tf.get_default_session(), self.path,
                        global_step=self.epoch_num, latest_filename='latest')
