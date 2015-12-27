#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: callback.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import sys
import numpy as np
import os
from abc import abstractmethod

from .naming import *

class Callback(object):
    def before_train(self):
        self.graph = tf.get_default_graph()
        self.sess = tf.get_default_session()
        self._before_train()

    def _before_train(self):
        """
        Called before starting iterative training
        """

    def trigger_step(self, inputs, outputs, cost):
        """
        Callback to be triggered after every step (every backpropagation)
        Args:
            inputs: the input dict fed into the graph
            outputs: list of output values after running this dp
            cost: the cost value after running this dp
        """

    def trigger_epoch(self):
        """
        Callback to be triggered after every epoch (full iteration of input dataset)
        """

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
    def __init__(self, log_dir, histogram_regex=None):
        self.log_dir = log_dir
        self.epoch_num = 0
        self.hist_regex = histogram_regex

    def _before_train(self):
        self.writer = tf.train.SummaryWriter(
            self.log_dir, graph_def=self.sess.graph_def)
        tf.add_to_collection(SUMMARY_WRITER_COLLECTION_KEY, self.writer)

        # create some summary
        if self.hist_regex is not None:
            import re
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for p in params:
                name = p.name
                if re.search(self.hist_regex, name):
                    tf.histogram_summary(name, p)

        self.summary_op = tf.merge_all_summaries()

    def trigger_step(self, inputs, outputs, cost):
        self.last_dp = inputs

    def trigger_epoch(self):
        # check if there is any summary
        if self.summary_op is None:
            return

        summary_str = self.summary_op.eval(self.last_dp)
        self.epoch_num += 1
        self.writer.add_summary(summary_str, self.epoch_num)

class Callbacks(Callback):
    def __init__(self, callbacks):
        # put SummaryWriter to the first
        for idx, cb in enumerate(callbacks):
            if type(cb) == SummaryWriter:
                callbacks.insert(0, callbacks.pop(idx))
                break
        else:
            raise RuntimeError("callbacks must contain a SummaryWriter!")

        self.callbacks = callbacks

    def before_train(self):
        for cb in self.callbacks:
            cb.before_train()

    def trigger_step(self, inputs, outputs, cost):
        for cb in self.callbacks:
            cb.trigger_step(inputs, outputs, cost)

    def trigger_epoch(self):
        for cb in self.callbacks:
            cb.trigger_epoch()

