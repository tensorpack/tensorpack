#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: steps.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" Some common step callbacks. """

import tensorflow as tf
import re
from six.moves import zip
import tqdm

from ..utils import logger, get_tqdm_kwargs
from ..utils.naming import (
    MOVING_SUMMARY_VARS_KEY,
    GLOBAL_STEP_INCR_VAR_NAME,
    LOCAL_STEP_OP_NAME)
from ..tfutils.common import get_op_tensor_name, get_global_step_var, get_global_step_value
from .base import Callback

__all__ = ['StepStatPrinter', 'MaintainStepCounter',
           'SummaryMovingAverage', 'ProgressBar']


class StepStatPrinter(Callback):
    """ It prints the value of some tensors in each step.
    It's just a demo of how trigger_step works but you should in general use
    :func:`symbolic_functions.print_stat` or :func:`tf.Print` instead. """

    def __init__(self, names):
        """
        Args:
            names(list): list of string, the names of the tensor to print.
        """
        names = [get_op_tensor_name(n)[1] for n in names]
        logger.warn("Using print_stat or tf.Print in the graph is much faster than StepStatPrinter!")
        self._names = names

    def _extra_fetches(self):
        return self._names

    def _trigger_step(self, *args):
        assert len(args) == len(self._names), len(args)
        for n, v in zip(self._names, args):
            logger.info("{}: {}".format(n, v))


class MaintainStepCounter(Callback):
    """
    It maintains the global step in the graph and also creates the local step tensor.
    This callback is always enabled by the trainer, and you wouldn't need to
    use it.
    """
    def _setup_graph(self):
        # ensure it exists
        get_global_step_var()
        self.gs_incr_var = self.trainer.sess.graph.get_tensor_by_name(GLOBAL_STEP_INCR_VAR_NAME)
        self.local_step = tf.mod(
            self.gs_incr_var, self.trainer.config.step_per_epoch,
            name=LOCAL_STEP_OP_NAME)

    def _before_train(self):
        gs_val = get_global_step_value()
        if gs_val != 0:
            logger.info("Start training with global_step={}".format(gs_val))

    def _extra_fetches(self):
        return [self.gs_incr_var.op]


class SummaryMovingAverage(Callback):
    """ Maintain the moving average of the tensors
        in every step, and summarize them. Enabled by default.
    """
    def __init__(self, collection=MOVING_SUMMARY_VARS_KEY, decay=0.95):
        """
        Args:
            collection(str): the collection of tensors to summarize. The
                default would work with :func:`add_moving_summary`.
            decay(float): the decay of the moving average.
        """
        self._collection = collection
        self._decay = decay

    def _setup_graph(self):
        tensors = set(tf.get_collection(self._collection))

        # TODO will produce tower0/xxx. not elegant
        with tf.name_scope(None):
            averager = tf.train.ExponentialMovingAverage(
                self._decay, num_updates=get_global_step_var(), name='EMA')
            avg_maintain_op = averager.apply(tensors)
            for idx, c in enumerate(tensors):
                name = re.sub('tower[p0-9]+/', '', c.op.name)
                tf.summary.scalar(name + '-summary', averager.average(c))
        self.ema_op = avg_maintain_op

    def _extra_fetches(self):
        return [self.ema_op]


class ProgressBar(Callback):
    """ A progress bar based on tqdm. Enabled by default. """
    def _before_train(self):
        self._total = self.trainer.config.step_per_epoch
        self._tqdm_args = get_tqdm_kwargs(leave=True)

    def _trigger_step(self, *args):
        if self.step_num == 0:
            self._bar = tqdm.trange(self._total, **self._tqdm_args)
        self._bar.update()
        if self.step_num == self._total - 1:
            self._bar.close()
