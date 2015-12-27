#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: validation_callback.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from .stat import *
from .callback import PeriodicCallback, Callback
from .naming import *
from .utils import *

class ValidationAccuracy(PeriodicCallback):
    """
    Validate the accuracy for the given correct and cost variable
    Use under the following setup:
        correct_var: integer, number of correct samples in this batch
        ds: batched dataset
    """
    def __init__(self, ds, prefix,
                 period=1,
                 correct_var_name='correct:0',
                 cost_var_name='cost:0'):
        super(ValidationAccuracy, self).__init__(period)
        self.ds = ds
        self.prefix = prefix

        self.correct_var_name = correct_var_name
        self.cost_var_name = cost_var_name

    def get_tensor(self, name):
        return self.graph.get_tensor_by_name(name)

    def _before_train(self):
        self.input_vars = tf.get_collection(INPUT_VARS_KEY)
        self.dropout_var = self.get_tensor(DROPOUT_PROB_VAR_NAME)
        self.correct_var = self.get_tensor(self.correct_var_name)
        self.cost_var = self.get_tensor(self.cost_var_name)
        self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]

    def _trigger(self):
        cnt = 0
        correct_stat = Accuracy()
        cost_sum = 0
        for dp in self.ds.get_data():
            feed = {self.dropout_var: 1.0}
            feed.update(dict(zip(self.input_vars, dp)))

            batch_size = dp[0].shape[0]   # assume batched input

            cnt += batch_size
            correct, cost = self.sess.run(
                [self.correct_var, self.cost_var], feed_dict=feed)
            correct_stat.feed(correct, batch_size)
            # each batch might not have the same size in validation
            cost_sum += cost * batch_size

        cost_avg = cost_sum / cnt
        self.writer.add_summary(
            create_summary('{}_accuracy'.format(self.prefix),
                           correct_stat.accuracy),
            self.epoch_num)
        self.writer.add_summary(
            create_summary('{}_cost'.format(self.prefix),
                           cost_avg),
            self.epoch_num)
        print "{} validation after epoch {}: acc={}, cost={}".format(
            self.prefix, self.epoch_num, correct_stat.accuracy, cost_avg)
