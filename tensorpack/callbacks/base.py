# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import sys
import os
import time
from abc import abstractmethod, ABCMeta

__all__ = ['Callback', 'PeriodicCallback', 'ProxyCallback']

class Callback(object):
    """ Base class for all callbacks """
    __metaclass__ = ABCMeta

    def before_train(self):
        """
        Called right before the first iteration.
        """
        self._before_train()

    def _before_train(self):
        pass

    def setup_graph(self, trainer):
        """
        Called before finalizing the graph.
        Use this callback to setup some ops used in the callback.

        :param trainer: :class:`train.Trainer` instance
        """
        self.trainer = trainer
        self.graph = tf.get_default_graph()
        self.epoch_num = self.trainer.config.starting_epoch - 1
        # self.epoch_num is always the number of epochs that finished updating parameters.
        with tf.name_scope(type(self).__name__):
            self._setup_graph()

    def _setup_graph(self):
        pass

    def after_train(self):
        """
        Called after training.
        """
        self._after_train()

    def _after_train(self):
        pass

    def trigger_step(self):
        """
        Callback to be triggered after every step (every backpropagation)

        Could be useful to apply some tricks on parameters (clipping, low-rank, etc)
        """

    def trigger_epoch(self):
        """
        Triggered after every epoch.

        In this function, self.epoch_num would be the number of epoch finished.
        """
        self.epoch_num += 1
        self._trigger_epoch()

    def _trigger_epoch(self):
        pass

    def __str__(self):
        return type(self).__name__

class ProxyCallback(Callback):
    def __init__(self, cb):
        self.cb = cb

    def _before_train(self):
        self.cb.before_train()

    def _setup_graph(self):
        self.cb.setup_graph(self.trainer)

    def _after_train(self):
        self.cb.after_train()

    def _trigger_epoch(self):
        self.cb.trigger_epoch()

    def __str__(self):
        return "Proxy-" + str(self.cb)

class PeriodicCallback(ProxyCallback):
    """
    A callback to be triggered after every `period` epochs.
    Doesn't work for trigger_step
    """
    def __init__(self, cb, period):
        """
        :param cb: a `Callback`
        :param period: int
        """
        super(PeriodicCallback, self).__init__(cb)
        self.period = int(period)

    def _trigger_epoch(self):
        if self.epoch_num % self.period == 0:
            self.cb.epoch_num = self.epoch_num - 1
            self.cb.trigger_epoch()

    def __str__(self):
        return "Periodic-" + str(self.cb)

