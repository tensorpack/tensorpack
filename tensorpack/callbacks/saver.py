# -*- coding: UTF-8 -*-
# File: saver.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from datetime import datetime
import os

from .base import Callback
from ..utils import logger
from ..utils.develop import log_deprecated
from ..tfutils.common import get_tf_version_number

__all__ = ['ModelSaver', 'MinSaver', 'MaxSaver']


class ModelSaver(Callback):
    """
    Save the model every epoch.
    """

    def __init__(self, max_to_keep=10,
                 keep_checkpoint_every_n_hours=0.5,
                 checkpoint_dir=None,
                 var_collections=tf.GraphKeys.GLOBAL_VARIABLES,
                 keep_recent=None, keep_freq=None):
        """
        Args:
            max_to_keep, keep_checkpoint_every_n_hours(int): the same as in ``tf.train.Saver``.
            checkpoint_dir (str): Defaults to ``logger.LOG_DIR``.
            var_collections (str or list of str): collection of the variables (or list of collections) to save.
        """
        self._max_to_keep = max_to_keep
        self._keep_every_n_hours = keep_checkpoint_every_n_hours
        if keep_recent is not None or keep_freq is not None:
            log_deprecated("ModelSaver(keep_recent=, keep_freq=)", "Use max_to_keep and keep_checkpoint_every_n_hours!")
            if keep_recent is not None:
                self._max_to_keep = keep_recent
            if keep_freq is not None:
                self._keep_every_n_hours = keep_freq

        if not isinstance(var_collections, list):
            var_collections = [var_collections]
        self.var_collections = var_collections
        if checkpoint_dir is None:
            checkpoint_dir = logger.LOG_DIR
        assert checkpoint_dir is not None
        if not tf.gfile.IsDirectory(checkpoint_dir):
            tf.gfile.MakeDirs(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

    def _setup_graph(self):
        vars = []
        for key in self.var_collections:
            vars.extend(tf.get_collection(key))
        vars = list(set(vars))
        self.path = os.path.join(self.checkpoint_dir, 'model')
        if get_tf_version_number() <= 1.1:
            self.saver = tf.train.Saver(
                var_list=vars,
                max_to_keep=self._max_to_keep,
                keep_checkpoint_every_n_hours=self._keep_every_n_hours,
                write_version=tf.train.SaverDef.V2)
        else:
            self.saver = tf.train.Saver(
                var_list=vars,
                max_to_keep=self._max_to_keep,
                keep_checkpoint_every_n_hours=self._keep_every_n_hours,
                write_version=tf.train.SaverDef.V2,
                save_relative_paths=True)
        # Don't know how it can be useful,
        # but since there is a predefined key, why not use it?
        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)

    def _before_train(self):
        # graph is finalized, OK to write it now.
        time = datetime.now().strftime('%m%d-%H%M%S')
        self.saver.export_meta_graph(
            os.path.join(self.checkpoint_dir,
                         'graph-{}.meta'.format(time)),
            collection_list=self.graph.get_all_collection_keys())

    def _trigger(self):
        try:
            self.saver.save(
                tf.get_default_session(),
                self.path,
                global_step=tf.train.get_global_step(),
                write_meta_graph=False)
            logger.info("Model saved to %s." % tf.train.get_checkpoint_state(self.checkpoint_dir).model_checkpoint_path)
        except (OSError, IOError, tf.errors.PermissionDeniedError,
                tf.errors.ResourceExhaustedError):   # disk error sometimes.. just ignore it
            logger.exception("Exception in ModelSaver!")


class MinSaver(Callback):
    """
    Separately save the model with minimum value of some statistics.
    """
    def __init__(self, monitor_stat, reverse=False, filename=None):
        """
        Args:
            monitor_stat(str): the name of the statistics.
            reverse (bool): if True, will save the maximum.
            filename (str): the name for the saved model.
                Defaults to ``min-{monitor_stat}.tfmodel``.

        Example:
            Save the model with minimum validation error to
            "min-val-error.tfmodel":

            .. code-block:: python

                MinSaver('val-error')

        Note:
            It assumes that :class:`ModelSaver` is used with
            ``checkpoint_dir=logger.LOG_DIR`` (the default). And it will save
            the model to that directory as well.
        """
        self.monitor_stat = monitor_stat
        self.reverse = reverse
        self.filename = filename
        self.min = None

    def _get_stat(self):
        try:
            v = self.trainer.monitors.get_latest(self.monitor_stat)
        except KeyError:
            v = None
        return v

    def _need_save(self):
        v = self._get_stat()
        if not v:
            return False
        return v > self.min if self.reverse else v < self.min

    def _trigger(self):
        if self.min is None or self._need_save():
            self.min = self._get_stat()
            if self.min:
                self._save()

    def _save(self):
        ckpt = tf.train.get_checkpoint_state(logger.LOG_DIR)
        if ckpt is None:
            raise RuntimeError(
                "Cannot find a checkpoint state. Do you forget to use ModelSaver?")
        path = ckpt.model_checkpoint_path

        newname = os.path.join(logger.LOG_DIR,
                               self.filename or
                               ('max-' + self.monitor_stat if self.reverse else 'min-' + self.monitor_stat))
        files_to_copy = tf.gfile.Glob(path + '*')
        for file_to_copy in files_to_copy:
            tf.gfile.Copy(file_to_copy, file_to_copy.replace(path, newname), overwrite=True)
        logger.info("Model with {} '{}' saved.".format(
            'maximum' if self.reverse else 'minimum', self.monitor_stat))


class MaxSaver(MinSaver):
    """
    Separately save the model with maximum value of some statistics.
    """
    def __init__(self, monitor_stat, filename=None):
        """
        Args:
            monitor_stat(str): the name of the statistics.
            filename (str): the name for the saved model.
                Defaults to ``max-{monitor_stat}.tfmodel``.
        """
        super(MaxSaver, self).__init__(monitor_stat, True, filename=filename)
