# -*- coding: UTF-8 -*-
# File: validation_callback.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import itertools
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from six.moves import zip

from ..utils import *
from ..utils.stat import *
from ..utils.summary import *
from .base import PeriodicCallback, Callback, TestCallback

__all__ = ['ValidationError', 'ValidationCallback', 'ValidationStatPrinter']

class ValidationCallback(PeriodicCallback):
    type = TestCallback()
    """
    Base class for validation callbacks.
    """
    def __init__(self, ds, prefix, period=1):
        super(ValidationCallback, self).__init__(period)
        self.ds = ds
        self.prefix = prefix

    def _before_train(self):
        self.input_vars = self.trainer.model.reuse_input_vars()
        self._find_output_vars()

    def get_tensor(self, name):
        return self.graph.get_tensor_by_name(name)

    @abstractmethod
    def _find_output_vars(self):
        """ prepare output variables. Will be called in before_train"""

    @abstractmethod
    def _get_output_vars(self):
        """ return a list of output vars to eval"""

    def _run_validation(self):
        """
        Eval the vars, generate inputs and outputs
        """
        output_vars = self._get_output_vars()
        sess = tf.get_default_session()
        with tqdm(total=self.ds.size(), ascii=True) as pbar:
            for dp in self.ds.get_data():
                feed = dict(zip(self.input_vars, dp))
                batch_size = dp[0].shape[0]   # assume batched input
                outputs = sess.run(output_vars, feed_dict=feed)
                yield (dp, outputs)
                pbar.update()

    @abstractmethod
    def _trigger_periodic(self):
        """ Implement the actual callback"""

class ValidationStatPrinter(ValidationCallback):
    """
    Write stat and summary of some Op for a validation dataset.
    The result of the given Op must be a scalar, and will be averaged for all batches in the validaion set.
    """
    def __init__(self, ds, names_to_print, prefix='validation', period=1):
        super(ValidationStatPrinter, self).__init__(ds, prefix, period)
        self.names = names_to_print

    def _find_output_vars(self):
        self.vars_to_print = [self.get_tensor(n) for n in self.names]

    def _get_output_vars(self):
        return self.vars_to_print

    def _trigger_periodic(self):
        stats = []
        for dp, outputs in self._run_validation():
            stats.append(outputs)
        stats = np.mean(stats, axis=0)
        assert len(stats) == len(self.vars_to_print)

        for stat, var in itertools.izip(stats, self.vars_to_print):
            name = var.name.replace(':0', '')
            self.trainer.summary_writer.add_summary(create_summary(
                '{}_{}'.format(self.prefix, name), stat), self.global_step)
            self.trainer.stat_holder.add_stat("{}_{}".format(self.prefix, name), stat)


class ValidationError(ValidationCallback):
    """
    Validate the accuracy from a 'wrong' variable
    wrong_var: integer, number of failed samples in this batch
    ds: batched dataset

    This callback produce the "true" error,
    taking account of the fact that batches might not have the same size in
    testing (because the size of test set might not be a multiple of batch size).
    In theory, the result could be different from what produced by ValidationStatPrinter.
    """
    def __init__(self, ds, prefix='validation',
                 period=1,
                 wrong_var_name='wrong:0'):
        super(ValidationError, self).__init__(ds, prefix, period)
        self.wrong_var_name = wrong_var_name

    def _find_output_vars(self):
        self.wrong_var = self.get_tensor(self.wrong_var_name)

    def _get_output_vars(self):
        return [self.wrong_var]

    def _trigger_periodic(self):
        err_stat = Accuracy()
        for dp, outputs in self._run_validation():
            batch_size = dp[0].shape[0]   # assume batched input
            wrong = outputs[0]
            err_stat.feed(wrong, batch_size)

        self.trainer.summary_writer.add_summary(create_summary(
            '{}_error'.format(self.prefix), err_stat.accuracy), self.global_step)
        self.trainer.stat_holder.add_stat("{}_error".format(self.prefix), err_stat.accuracy)
