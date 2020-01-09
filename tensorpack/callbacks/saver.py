# -*- coding: utf-8 -*-
# File: saver.py


import os
from datetime import datetime

from ..compat import tfv1 as tf
from ..utils import fs, logger
from .base import Callback

__all__ = ['ModelSaver', 'MinSaver', 'MaxSaver']


class ModelSaver(Callback):
    """
    Save the model once triggered.
    """

    def __init__(self, max_to_keep=10,
                 keep_checkpoint_every_n_hours=0.5,
                 checkpoint_dir=None,
                 var_collections=None):
        """
        Args:
            max_to_keep (int): the same as in ``tf.train.Saver``.
            keep_checkpoint_every_n_hours (float): the same as in ``tf.train.Saver``.
                Note that "keep" does not mean "create", but means "don't delete".
            checkpoint_dir (str): Defaults to ``logger.get_logger_dir()``.
            var_collections (str or list of str): collection of the variables (or list of collections) to save.
        """
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        self._max_to_keep = max_to_keep
        self._keep_every_n_hours = keep_checkpoint_every_n_hours

        if not isinstance(var_collections, list):
            var_collections = [var_collections]
        self.var_collections = var_collections
        if checkpoint_dir is None:
            checkpoint_dir = logger.get_logger_dir()
        if checkpoint_dir is not None:
            if not tf.gfile.IsDirectory(checkpoint_dir):  # v2: tf.io.gfile.isdir
                tf.gfile.MakeDirs(checkpoint_dir)  # v2: tf.io.gfile.makedirs
        # If None, allow it to be init, but fail later if used
        # For example, if chief_only=True, it can still be safely initialized
        # in non-chief workers which don't have logger dir
        self.checkpoint_dir = fs.normpath(checkpoint_dir) if checkpoint_dir is not None else checkpoint_dir

    def _setup_graph(self):
        assert self.checkpoint_dir is not None, \
            "Please provide 'checkpoint_dir' for ModelSaver, or use logger.set_logger_dir()"
        vars = []
        for key in self.var_collections:
            vars.extend(tf.get_collection(key))
        vars = list(set(vars))
        self.path = os.path.join(self.checkpoint_dir, 'model')
        self.saver = tf.train.Saver(
            var_list=vars,
            max_to_keep=self._max_to_keep,
            keep_checkpoint_every_n_hours=self._keep_every_n_hours,
            write_version=tf.train.SaverDef.V2,
            save_relative_paths=True)
        # Scaffold will call saver.build from this collection
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
        except (IOError, tf.errors.PermissionDeniedError,
                tf.errors.ResourceExhaustedError):   # disk error sometimes.. just ignore it
            logger.exception("Exception in ModelSaver!")


class MinSaver(Callback):
    """
    Separately save the model with minimum value of some statistics.
    """
    def __init__(self, monitor_stat, reverse=False, filename=None, checkpoint_dir=None):
        """
        Args:
            monitor_stat(str): the name of the statistics.
            reverse (bool): if True, will save the maximum.
            filename (str): the name for the saved model.
                Defaults to ``min-{monitor_stat}.tfmodel``.
            checkpoint_dir (str): the directory containing checkpoints.

        Example:
            Save the model with minimum validation error to
            "min-val-error.tfmodel":

            .. code-block:: python

                MinSaver('val-error')

        Note:
            1. It assumes that :class:`ModelSaver` is used with the same ``checkpoint_dir``
               and appears earlier in the callback list.
               The default for both :class:`ModelSaver` and :class:`MinSaver`
               is ``checkpoint_dir=logger.get_logger_dir()``
            2. Callbacks are executed in the order they are defined. Therefore you'd want to
               use this callback after the callback (e.g. InferenceRunner) that produces the statistics.
        """
        self.monitor_stat = monitor_stat
        self.reverse = reverse
        self.filename = filename
        self.best = None
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            self.checkpoint_dir = logger.get_logger_dir()
        self.checkpoint_dir = fs.normpath(self.checkpoint_dir)

    def _get_stat(self):
        try:
            v = self.trainer.monitors.get_history(self.monitor_stat)[-1]
        except (KeyError, IndexError):
            v = None, None
        return v

    def _trigger(self):
        curr_step, curr_val = self._get_stat()
        if curr_step is None:
            return

        if self.best is None or (curr_val > self.best[1] if self.reverse else curr_val < self.best[1]):
            self.best = (curr_step, curr_val)
            self._save()

    def _save(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt is None:
            raise RuntimeError(
                "[MinSaver] Cannot find a checkpoint state. Do you forget to use ModelSaver?")
        path = ckpt.model_checkpoint_path

        extreme_name = 'maximum' if self.reverse else 'minimum'
        if not path.endswith(str(self.best[0])):
            logger.warn("[MinSaver] New {} '{}' found at global_step={}, but the latest checkpoint is {}.".format(
                extreme_name, self.monitor_stat, self.best[0], path
            ))
            logger.warn("MinSaver will do nothing this time. "
                        "The callbacks may have inconsistent frequency or wrong order.")
            return

        newname = os.path.join(self.checkpoint_dir,
                               self.filename or
                               ('max-' + self.monitor_stat if self.reverse else 'min-' + self.monitor_stat))
        files_to_copy = tf.gfile.Glob(path + '*')
        for file_to_copy in files_to_copy:
            tf.gfile.Copy(file_to_copy, file_to_copy.replace(path, newname), overwrite=True)
        logger.info("Model at global_step={} with {} {}={:.5g} saved.".format(
            self.best[0], extreme_name, self.monitor_stat, self.best[1]))


class MaxSaver(MinSaver):
    """
    Separately save the model with maximum value of some statistics.

    See docs of :class:`MinSaver` for details.
    """
    def __init__(self, monitor_stat, filename=None, checkpoint_dir=None):
        """
        Args:
            monitor_stat(str): the name of the statistics.
            filename (str): the name for the saved model.
                Defaults to ``max-{monitor_stat}.tfmodel``.
        """
        super(MaxSaver, self).__init__(monitor_stat, True, filename=filename, checkpoint_dir=checkpoint_dir)
