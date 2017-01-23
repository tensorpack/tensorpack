#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: steps.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" Some common step callbacks. """

import tensorflow as tf
import re
from six.moves import zip

from ..utils import logger
from ..utils.naming import MOVING_SUMMARY_VARS_KEY
from ..tfutils.common import get_op_tensor_name, get_global_step_var
from .base import Callback

__all__ = ['StepStatPrinter', 'SummaryMovingAverage']


class StepStatPrinter(Callback):
    """ It prints the value of some tensors in each step.
    It's just a demo of how trigger_step works but you should in general use
    :func:`print_stat` or :func:`tf.Print` instead. """

    def __init__(self, names):
        names = [get_op_tensor_name(n)[1] for n in names]
        logger.warn("Using print_stat or tf.Print in the graph is much faster than StepStatPrinter!")
        self._names = names

    def _extra_fetches(self):
        return self._names

    def _trigger_step(self, *args):
        assert len(args) == len(self._names), len(args)
        for n, v in zip(self._names, args):
            logger.info("{}: {}".format(n, v))


class SummaryMovingAverage(Callback):
    """ Maintain the moving average of the tensors
        in every step, and summarize them.
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
