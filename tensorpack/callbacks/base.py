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

__all__ = ['Callback', 'PeriodicCallback']

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

    def after_train(self):
        """
        Called after training
        """

    def trigger_step(self):
        """
        Callback to be triggered after every step (every backpropagation)
        Could be useful to apply some tricks on parameters (clipping, low-rank, etc)
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
            self.global_step = get_global_step()
            self._trigger()

    @abstractmethod
    def _trigger(self):
        pass

