#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: steps.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" Some common step callbacks. """

import tensorflow as tf
from six.moves import zip
import tqdm

from ..utils import logger, get_tqdm_kwargs
from ..utils.naming import GLOBAL_STEP_INCR_OP_NAME
from ..tfutils.common import (
    get_op_tensor_name, get_global_step_var,
    get_global_step_value, get_op_or_tensor_by_name)
from .base import Callback

__all__ = ['StepTensorPrinter', 'MaintainStepCounter',
           'ProgressBar']


class StepTensorPrinter(Callback):
    """ It prints the value of some tensors in each step.
    It's just a demo of how trigger_step works but you should in general use
    :func:`symbolic_functions.print_stat` or :func:`tf.Print` instead. """

    def __init__(self, names):
        """
        Args:
            names(list): list of string, the names of the tensors to print.
        """
        names = [get_op_tensor_name(n)[1] for n in names]
        logger.warn("Using print_stat or tf.Print in the graph is much faster than StepTensorPrinter!")
        self._names = names

    def _before_train(self):
        self._fetches = get_op_or_tensor_by_name(self._names)

    def _before_run(self, _):
        return self._fetches

    def _after_run(self, _, vals):
        args = vals.results
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
        gs_var = get_global_step_var()
        with tf.name_scope(None):
            self.gs_incr_var = tf.assign_add(
                gs_var, 1,
                name=GLOBAL_STEP_INCR_OP_NAME)
            # tf.mod(
            #     self.gs_incr_var, self.trainer.config.steps_per_epoch,
            #     name=LOCAL_STEP_OP_NAME)
        self._fetches = tf.train.SessionRunArgs(self.gs_incr_var)

    def _before_train(self):
        gs_val = get_global_step_value()
        if gs_val != 0:
            logger.info("Start training with global_step={}".format(gs_val))
        self._last_updated = self.trainer.local_step

    def _before_run(self, _):
        # increase global_step, when trainer.local_step changed
        if self.trainer.local_step != self._last_updated:
            self._last_updated = self.trainer.local_step
            return self._fetches
        else:
            return None


class ProgressBar(Callback):
    """ A progress bar based on tqdm. Enabled by default. """

    def __init__(self, names=[]):
        """
        Args:
            names(list): list of string, the names of the tensors to display.
        """
        super(ProgressBar, self).__init__()
        self._names = [get_op_tensor_name(n)[1] for n in names]
        self._tags = [get_op_tensor_name(n)[0].split("/")[-1] for n in names]

    def _before_train(self):
        self._last_updated = self.trainer.local_step

        self._total = self.trainer.config.steps_per_epoch
        self._tqdm_args = get_tqdm_kwargs(leave=True)

        self._fetches = get_op_or_tensor_by_name(self._names) or None
        if self._fetches:
            self._fetches = tf.train.SessionRunArgs(self._fetches)
            self._tqdm_args['bar_format'] = self._tqdm_args['bar_format'] + "{postfix} "

    def _before_run(self, _):
        if self.trainer.local_step != self._last_updated:
            self._last_updated = self.trainer.local_step

            if self.trainer.local_step == 0:
                self._bar = tqdm.trange(self._total, **self._tqdm_args)

            return self._fetches
        else:
            return None

    def _after_run(self, _, run_values):
        res = run_values.results
        if res:
            self._bar.set_postfix(zip(self._tags, res))

    def _trigger_step(self):
        self._bar.update()
        if self.trainer.local_step == self._total - 1:
            self._bar.close()

    def _after_train(self):
        self._bar.close()
