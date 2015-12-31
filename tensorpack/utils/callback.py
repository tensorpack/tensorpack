#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: callback.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import sys
import numpy as np
import os
import time
from abc import abstractmethod, ABCMeta

from . import create_test_session
from .naming import *
import logger

class Callback(object):
    __metaclass__ = ABCMeta
    running_graph = 'train'
    """ The graph that this callback should run on.
        Either 'train' or 'test'
    """

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
            inputs: the list of input values
            outputs: list of output values after running this inputs
            cost: the cost value after running this input
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
    def __init__(self, period=1, keep_recent=50, keep_freq=0.5):
        super(PeriodicSaver, self).__init__(period)
        self.path = os.path.join(logger.LOG_DIR, 'model')
        self.keep_recent = keep_recent
        self.keep_freq = keep_freq

    def _before_train(self):
        self.saver = tf.train.Saver(
            max_to_keep=self.keep_recent,
            keep_checkpoint_every_n_hours=self.keep_freq)

    def _trigger(self):
        self.saver.save(tf.get_default_session(), self.path,
                        global_step=self.epoch_num)

class SummaryWriter(Callback):
    def __init__(self):
        self.log_dir = logger.LOG_DIR
        self.epoch_num = 0

    def _before_train(self):
        self.writer = tf.train.SummaryWriter(
            self.log_dir, graph_def=self.sess.graph_def)
        tf.add_to_collection(SUMMARY_WRITER_COLLECTION_KEY, self.writer)
        self.summary_op = tf.merge_all_summaries()

    def trigger_epoch(self):
        # check if there is any summary
        if self.summary_op is None:
            return
        summary_str = self.summary_op.eval()
        self.epoch_num += 1
        self.writer.add_summary(summary_str, self.epoch_num)


class CallbackTimeLogger(object):
    def __init__(self):
        self.times = []
        self.tot = 0

    def add(self, name, time):
        self.tot += time
        self.times.append((name, time))

    def log(self):
        """ log the time of some heavy callbacks """
        if self.tot < 3:
            return
        msgs = []
        for name, t in self.times:
            if t / self.tot > 0.3 and t > 1:
                msgs.append("{}:{}".format(name, t))
        logger.info(
            "Callbacks took {} sec. {}".format(
                self.tot, ' '.join(msgs)))


class TrainCallbacks(Callback):
    def __init__(self, callbacks):
        self.cbs = callbacks
        # put SummaryWriter to the first
        for idx, cb in enumerate(self.cbs):
            if type(cb) == SummaryWriter:
                self.cbs.insert(0, self.cbs.pop(idx))
                break
        else:
            raise RuntimeError("Callbacks must contain a SummaryWriter!")

    def before_train(self):
        for cb in self.cbs:
            cb.before_train()
        self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]

    def trigger_step(self, inputs, outputs, cost):
        for cb in self.cbs:
            cb.trigger_step(inputs, outputs, cost)

    def trigger_epoch(self):
        tm = CallbackTimeLogger()
        for cb in self.cbs:
            s = time.time()
            cb.trigger_epoch()
            tm.add(type(cb).__name__, time.time() - s)
        self.writer.flush()
        tm.log()

class TestCallbacks(Callback):
    """
    Hold callbacks to be run in testing graph.
    Will set a context with testing graph and testing session, for
    each test-time callback to run
    """
    def __init__(self, callbacks):
        self.cbs = callbacks

    def before_train(self):
        self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]
        with create_test_session() as sess:
            self.sess = sess
            self.graph = sess.graph

            self.saver = tf.train.Saver()
            tf.add_to_collection(SUMMARY_WRITER_COLLECTION_KEY, self.writer)
            for cb in self.cbs:
                cb.before_train()

    def trigger_epoch(self):
        if not self.cbs:
            return
        tm = CallbackTimeLogger()
        with self.graph.as_default(), self.sess.as_default():
            s = time.time()
            ckpt = tf.train.get_checkpoint_state(logger.LOG_DIR)
            if ckpt is None:
                logger.error(
                    "Cannot find a checkpoint state. Do you forget to use PeriodicSaver?")
                return
            logger.info(
                "Restore checkpoint from {}".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            tm.add('restore session', time.time() - s)
            for cb in self.cbs:
                s = time.time()
                cb.trigger_epoch()
                tm.add(type(cb).__name__, time.time() - s)
        self.writer.flush()
        tm.log()

class Callbacks(Callback):
    def __init__(self, cbs):
        train_cbs = []
        test_cbs = []
        for cb in cbs:
            assert isinstance(cb, Callback), cb.__class__
            if cb.running_graph == 'test':
                test_cbs.append(cb)
            elif cb.running_graph == 'train':
                train_cbs.append(cb)
            else:
                raise RuntimeError(
                    "Unknown callback running graph {}!".format(cb.running_graph))
        self.train = TrainCallbacks(train_cbs)
        self.test = TestCallbacks(test_cbs)

    def before_train(self):
        self.train.before_train()
        self.test.before_train()

    def trigger_step(self, inputs, outputs, cost):
        self.train.trigger_step(inputs, outputs, cost)
        # test callback don't have trigger_step

    def trigger_epoch(self):
        self.train.trigger_epoch()
        self.test.trigger_epoch()

