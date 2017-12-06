#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: steps.py

""" Some common step callbacks. """

import tensorflow as tf
from six.moves import zip
import tqdm

from ..utils import logger
from ..utils.utils import get_tqdm_kwargs
from ..utils.naming import GLOBAL_STEP_INCR_OP_NAME
from ..tfutils.common import (
    get_op_tensor_name, get_global_step_var)
from .base import Callback

__all__ = ['TensorPrinter', 'StepTensorPrinter', 'ProgressBar']


class TensorPrinter(Callback):
    """ Prints the value of some tensors in each step.
        It's an example of how ``before_run/after_run`` works.
    """

    def __init__(self, names):
        """
        Args:
            names(list): list of string, the names of the tensors to print.
        """
        names = [get_op_tensor_name(n)[1] for n in names]
        logger.warn("Using print_stat or tf.Print in the graph is much faster than StepTensorPrinter!")
        self._names = names

    def _setup_graph(self):
        self._fetches = self.get_tensors_maybe_in_tower(self._names)

    def _before_run(self, _):
        return self._fetches

    def _after_run(self, _, vals):
        args = vals.results
        assert len(args) == len(self._names), len(args)
        for n, v in zip(self._names, args):
            logger.info("{}: {}".format(n, v))


StepTensorPrinter = TensorPrinter


class ProgressBar(Callback):
    """ A progress bar based on tqdm. Enabled by default. """

    _chief_only = False

    def __init__(self, names=[]):
        """
        Args:
            names(list): list of string, the names of the tensors to monitor
                on the progress bar.
        """
        super(ProgressBar, self).__init__()
        self._names = [get_op_tensor_name(n)[1] for n in names]
        self._tags = [get_op_tensor_name(n)[0].split("/")[-1] for n in names]
        self._bar = None

    def _before_train(self):
        self._last_updated = self.local_step

        self._total = self.trainer.steps_per_epoch
        self._tqdm_args = get_tqdm_kwargs(leave=True)

        self._fetches = self.get_tensors_maybe_in_tower(self._names) or None
        if self._fetches:
            for t in self._fetches:
                assert t.shape.ndims == 0, "ProgressBar can only print scalars, not {}".format(t)
            self._fetches = tf.train.SessionRunArgs(self._fetches)
            self._tqdm_args['bar_format'] = self._tqdm_args['bar_format'] + "{postfix} "

    def _before_epoch(self):
        self._bar = tqdm.trange(self._total, **self._tqdm_args)

    def _after_epoch(self):
        self._bar.close()

    def _before_run(self, _):
        # update progress bar when local step changed (one step is finished)
        if self.local_step != self._last_updated:
            self._last_updated = self.local_step
            return self._fetches
        else:
            return None

    def _after_run(self, _, run_values):
        res = run_values.results
        if res:
            self._bar.set_postfix(zip(self._tags, res))

    def _trigger_step(self):
        self._bar.update()

    def _after_train(self):
        if self._bar:       # training may get killed before the first step
            self._bar.close()


class MaintainStepCounter(Callback):
    """
    It maintains the global step in the graph, making sure it's increased by one.
    This callback is used by the trainer, you don't need to worry about it.
    """

    _chief_only = False
    """
    In distributed training, we let each worker maintain its local global_step.
    """

    def _setup_graph(self):
        # ensure it exists
        gs_var = get_global_step_var()
        with tf.name_scope(None):
            with self.graph.colocate_with(gs_var):
                self.gs_incr_op = tf.assign_add(
                    gs_var, 1,
                    name=GLOBAL_STEP_INCR_OP_NAME).op
        self._fetches = tf.train.SessionRunArgs(self.gs_incr_op)

    def _before_train(self):
        if self.global_step != 0:
            logger.info("Start training with global_step={}".format(self.global_step))

    def _before_run(self, _):
        # always increase global_step when hooked_sess.run is called
        return self._fetches

    def _after_run(self, _, __):
        # Keep python-side global_step in agreement with TF-side
        self.trainer.loop._global_step += 1
