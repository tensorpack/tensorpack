# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import os
import re

from .base import Callback
from ..utils import *

__all__ = ['ModelSaver']

class ModelSaver(Callback):
    """
    Save the model to logger directory.
    """
    def __init__(self, keep_recent=10, keep_freq=0.5):
        """
        :param keep_recent: see `tf.train.Saver` documentation.
        :param keep_freq: see `tf.train.Saver` documentation.
        """
        self.keep_recent = keep_recent
        self.keep_freq = keep_freq

    def _before_train(self):
        self.path = os.path.join(logger.LOG_DIR, 'model')
        self.saver = tf.train.Saver(
            var_list=ModelSaver._get_vars(),
            max_to_keep=self.keep_recent,
            keep_checkpoint_every_n_hours=self.keep_freq)

    @staticmethod
    def _get_vars():
        vars = tf.all_variables()
        var_dict = {}
        for v in vars:
            name = v.op.name
            if re.match('tower[1-9]', name):
                #logger.info("Skip {} when saving model.".format(name))
                continue
            if 'tower0/' in name:
                new_name = name.replace('tower0/', '')
                logger.info(
                    "{} renamed to {} when saving model.".format(name, new_name))
                name = new_name
            var_dict[name] = v
        return var_dict

    def _trigger_epoch(self):
        self.saver.save(
            tf.get_default_session(),
            self.path,
            global_step=self.global_step)

class MinSaver(Callback):
    def __init__(self, monitor_stat):
        self.monitor_stat = monitor_stat

    def _trigger_epoch(self):
        pass
