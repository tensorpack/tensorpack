#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: steps.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" Some common step callbacks. """

from six.moves import zip
from ..utils import logger
from ..tfutils.common import get_op_tensor_name
from ..tfutils.summary import summary_moving_average
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
        for n, v in zip(self._names, args):
            logger.info("{}: {}".format(n, v))


class SummaryMovingAverage(Callback):
    """ Maintain the moving average of the tensors added by :func:`summary.add_moving_summary`
        in every step, and summarize them.
    """
    def _setup_graph(self):
        self.ema_op = summary_moving_average()

    def _extra_fetches(self):
        return [self.ema_op]
