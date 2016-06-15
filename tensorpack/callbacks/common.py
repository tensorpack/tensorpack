# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import os, shutil
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

    def _setup_graph(self):
        self.path = os.path.join(logger.LOG_DIR, 'model')
        self.saver = tf.train.Saver(
            var_list=ModelSaver._get_vars(),
            max_to_keep=self.keep_recent,
            keep_checkpoint_every_n_hours=self.keep_freq)
        self.meta_graph_written = False

    @staticmethod
    def _get_vars():
        vars = tf.all_variables()
        var_dict = {}
        for v in vars:
            name = v.op.name
            if re.match('tower[p1-9]', name):
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
            global_step=self.global_step,
            write_meta_graph=not self.meta_graph_written)
        if not self.meta_graph_written:
            self.meta_graph_written = True

class MinSaver(Callback):
    def __init__(self, monitor_stat):
        self.monitor_stat = monitor_stat
        self.min = None

    def _get_stat(self):
        return self.trainer.stat_holder.get_stat_now(self.monitor_stat)

    def _trigger_epoch(self):
        if self.min is None or self._get_stat() < self.min:
            self.min = self._get_stat()
            self._save()

    def _save(self):
        ckpt = tf.train.get_checkpoint_state(logger.LOG_DIR)
        if ckpt is None:
            raise RuntimeError(
                "Cannot find a checkpoint state. Do you forget to use ModelSaver before MinSaver?")
        path = chpt.model_checkpoint_path
        newname = os.path.join(logger.LOG_DIR, 'min_' + self.monitor_stat)
        shutil.copy(path, newname)
        logger.info("Model with minimum {} saved.".format(self.monitor_stat))


