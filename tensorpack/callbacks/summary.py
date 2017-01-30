#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import re

from ..utils.naming import MOVING_SUMMARY_VARS_KEY
from ..tfutils.common import get_global_step_var
from .base import Callback

__all__ = ['MovingAverageSummary']


class MovingAverageSummary(Callback):
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
