#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import sys
import os
import time
from abc import abstractmethod, ABCMeta

from ..utils import *

__all__ = ['Callback', 'PeriodicCallback', 'TrainCallback', 'TestCallback']

class TrainCallback(object):
    pass

class TestCallback(object):
    pass

class Callback(object):
    __metaclass__ = ABCMeta

    type = TrainCallback()
    """ The graph that this callback should run on.
        Either TrainCallback or TestCallback
    """

    def before_train(self, trainer):
        self.trainer = trainer
        self.graph = tf.get_default_graph()
        self.sess = tf.get_default_session()
        self.epoch_num = 0
        self._before_train()

    def _before_train(self):
        """
        Called before starting iterative training
        """

    def after_train(self):
        self._after_train()

    def _after_train(self):
        """
        Called after training
        """

    def trigger_step(self):
        """
        Callback to be triggered after every step (every backpropagation)
        Could be useful to apply some tricks on parameters (clipping, low-rank, etc)
        """

    @property
    def global_step(self):
        return self.trainer.global_step

    def trigger_epoch(self):
        self.epoch_num += 1
        self._trigger_epoch()

    def _trigger_epoch(self):
        """
        Callback to be triggered after every epoch (full iteration of input dataset)
        """


class PeriodicCallback(Callback):
    def __init__(self, period):
        self.period = period

    def _trigger_epoch(self):
        if self.epoch_num % self.period == 0:
            self._trigger_periodic()

    @abstractmethod
    def _trigger_periodic(self):
        pass

