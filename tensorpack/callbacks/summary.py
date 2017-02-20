#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..utils.naming import MOVING_SUMMARY_OPS_KEY
from .base import Callback

__all__ = ['MovingAverageSummary']


class MovingAverageSummary(Callback):
    """ Maintain the moving average of the tensors
        in every step, and summarize them. Enabled by default.
    """
    def __init__(self, collection=MOVING_SUMMARY_OPS_KEY):
        """
        Args:
            collection(str): the collection of EMA-maintaining ops.
                The default would work with :func:`add_moving_summary()`,
                but you can use some others.
        """
        self._collection = collection

    def _setup_graph(self):
        ops = tf.get_collection(self._collection)
        self.ema_op = tf.group(*ops, name='summary_moving_averages')

    def _before_run(self, _):
        return [self.ema_op]
