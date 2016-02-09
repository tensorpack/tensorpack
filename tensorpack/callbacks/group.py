#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: group.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from contextlib import contextmanager

from .base import Callback
from .common import *
from ..utils import *

__all__ = ['Callbacks']

@contextmanager
def create_test_graph():
    G = tf.get_default_graph()
    model = G.get_collection(MODEL_KEY)[0]
    with tf.Graph().as_default() as Gtest:
        # create a global step var in test graph
        global_step_var = tf.Variable(
            0, trainable=False, name=GLOBAL_STEP_OP_NAME)
        new_model = model.__class__()
        input_vars = new_model.get_input_vars()
        cost = new_model.get_cost(input_vars, is_training=False)
        Gtest.add_to_collection(MODEL_KEY, new_model)
        yield Gtest

@contextmanager
def create_test_session():
    with create_test_graph():
        with tf.Session() as sess:
            yield sess

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
                msgs.append("{}:{:.3f}sec".format(name, t))
        logger.info(
            "Callbacks took {:.3f} sec in total. {}".format(
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
            raise ValueError("Callbacks must contain a SummaryWriter!")

    def _before_train(self):
        for cb in self.cbs:
            cb.before_train()
        self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]

    def _after_train(self):
        for cb in self.cbs:
            cb.after_train()

    def trigger_step(self):
        for cb in self.cbs:
            cb.trigger_step()

    def _trigger_epoch(self):
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

    def _before_train(self):
        self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]
        with create_test_session() as sess:
            self.sess = sess
            self.graph = sess.graph

            self.saver = tf.train.Saver()
            tf.add_to_collection(SUMMARY_WRITER_COLLECTION_KEY, self.writer)
            for cb in self.cbs:
                cb.before_train()

    def _after_train(self):
        for cb in self.cbs:
            cb.after_train()

    def _trigger_epoch(self):
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
                raise ValueError(
                    "Unknown callback running graph {}!".format(cb.running_graph))
        self.train = TrainCallbacks(train_cbs)
        if test_cbs:
            self.test = TestCallbacks(test_cbs)
        else:
            self.test = None

    def _before_train(self):
        self.train.before_train()
        if self.test:
            self.test.before_train()

    def _after_train(self):
        self.train.after_train()
        if self.test:
            self.test.after_train()

    def trigger_step(self):
        self.train.trigger_step()
        # test callback don't have trigger_step

    def _trigger_epoch(self):
        self.train.trigger_epoch()
        # TODO test callbacks can be run async?
        if self.test:
            self.test.trigger_epoch()
