#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: callback.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import sys
import numpy as np
import os
from abc import abstractmethod
from .stat import *
from .utils import *
from .naming import *

class Callback(object):
    def before_train(self):
        self.graph = tf.get_default_graph()
        self.sess = tf.get_default_session()
        self._before_train()

    def _before_train(self):
        """
        Called before training
        """

    # trigger after every step
    def trigger_step(self, dp, outputs, cost):
        """
        Args:
            dp: the input dict fed into the graph
            outputs: list of output values after running this dp
            cost: the cost value after running this dp
        """
        pass

    # trigger after every epoch
    def trigger_epoch(self):
        pass

class PeriodicCallback(Callback):
    def __init__(self, period):
        self.__period = period
        self.epoch_num = 0

    def trigger_epoch(self):
        self.epoch_num += 1
        if self.epoch_num % self.__period == 0:
            self._trigger()

    @abstractmethod
    def _trigger(self):
        pass

class AccuracyValidation(PeriodicCallback):
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
        super(AccuracyValidation, self).__init__(period)
        self.ds = ds
        self.prefix = prefix

        self.correct_var_name = correct_var_name
        self.cost_var_name = cost_var_name

    def get_tensor(self, name):
        return self.graph.get_tensor_by_name(name)

    def _before_train(self):
        self.input_vars = self.graph.get_collection(INPUT_VARS_KEY)
        self.dropout_var = self.get_tensor(DROPOUT_PROB_VAR_NAME)
        self.correct_var = self.get_tensor(self.correct_var_name)
        self.cost_var = self.get_tensor(self.cost_var_name)
        try:
            self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]
        except Exception as e:
            print "SummaryWriter should be the first extension!"
            raise

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
            create_summary('{} accuracy'.format(self.prefix),
                           correct_stat.accuracy),
            self.epoch_num)
        self.writer.add_summary(
            create_summary('{} cost'.format(self.prefix),
                           cost_avg),
            self.epoch_num)
        print "{} validation after epoch {}: acc={}, cost={}".format(
            self.prefix, self.epoch_num, correct_stat.accuracy, cost_avg)

class TrainingAccuracy(Callback):
    def __init__(self, correct_var_name='correct:0'):
        """
            correct_var: number of correct sample in this batch
        """
        self.correct_var_name = correct_var_name
        self.epoch_num = 0

    def _before_train(self):
        try:
            self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]
        except Exception as e:
            print "SummaryWriter should be the first extension!"
            raise
        output_vars = self.graph.get_collection(OUTPUT_VARS_KEY)
        for idx, var in enumerate(output_vars):
            if var.name == self.correct_var_name:
                self.correct_output_idx = idx
                break
        else:
            raise RuntimeError(
                "'correct' variable must be in the model outputs to use TrainingAccuracy")
        self.running_cost = StatCounter()
        self.running_acc = Accuracy()

    def trigger_step(self, inputs, outputs, cost):
        self.running_cost.feed(cost)
        self.running_acc.feed(
            outputs[self.correct_output_idx],
            inputs[0].shape[0]) # assume batch input

    def trigger_epoch(self):
        self.epoch_num += 1
        print('Training average in Epoch {}: cost={}, acc={}'.format
              (self.epoch_num, self.running_cost.average,
              self.running_acc.accuracy))
        self.writer.add_summary(
            create_summary('training average accuracy', self.running_acc.accuracy),
            self.epoch_num)
        self.writer.add_summary(
            create_summary('training average cost', self.running_cost.average),
            self.epoch_num)

        self.running_cost.reset()
        self.running_acc.reset()

class PeriodicSaver(PeriodicCallback):
    def __init__(self, log_dir, period=1):
        super(PeriodicSaver, self).__init__(period)
        self.path = os.path.join(log_dir, 'model')

    def _before_train(self):
        self.saver = tf.train.Saver(max_to_keep=99999)

    def _trigger(self):
        self.saver.save(tf.get_default_session(), self.path,
                        global_step=self.epoch_num, latest_filename='latest')

class SummaryWriter(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.epoch_num = 0

    def _before_train(self):
        sess = tf.get_default_session()
        graph = tf.get_default_graph()
        self.writer = tf.train.SummaryWriter(
            self.log_dir, graph_def=sess.graph_def)
        graph.add_to_collection(SUMMARY_WRITER_COLLECTION_KEY, self.writer)
        self.summary_op = tf.merge_all_summaries()

    def trigger_step(self, dp, outputs, cost):
        self.last_dp = dp

    def trigger_epoch(self):
        # check if there is any summary
        if self.summary_op is None:
            return
        summary_str = self.summary_op.eval(self.last_dp)
        self.epoch_num += 1
        self.writer.add_summary(summary_str, self.epoch_num)

