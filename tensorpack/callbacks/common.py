# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import os
import re

from .base import Callback, PeriodicCallback
from ..utils import *

__all__ = ['PeriodicSaver']

class PeriodicSaver(PeriodicCallback):
    def __init__(self, period=1, keep_recent=10, keep_freq=0.5):
        super(PeriodicSaver, self).__init__(period)
        self.keep_recent = keep_recent
        self.keep_freq = keep_freq

    def _before_train(self):
        self.path = os.path.join(logger.LOG_DIR, 'model')
        self.saver = tf.train.Saver(
            max_to_keep=self.keep_recent,
            keep_checkpoint_every_n_hours=self.keep_freq)

    def _trigger_periodic(self):
        self.saver.save(
            tf.get_default_session(),
            self.path,
            global_step=self.global_step)

class MinSaver(Callback):
    def __init__(self, monitor_stat):
        self.monitor_stat = monitor_stat

    def _trigger_epoch(self):
        pass
