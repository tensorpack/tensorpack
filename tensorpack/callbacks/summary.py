#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..utils.naming import MOVING_SUMMARY_OPS_KEY
from .base import Callback

__all__ = ['MovingAverageSummary', 'MergeAllSummaries']


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


class MergeAllSummaries_RunAlone(Callback):
    def __init__(self, period, key):
        self._period = period
        self._key = key

    def _setup_graph(self):
        self.summary_op = tf.summary.merge_all(self._key)

    def _trigger_step(self):
        if self._period:
            if (self.local_step + 1) % self._period == 0:
                self._trigger()

    def _trigger(self):
        if self.summary_op:
            summary = self.summary_op.eval()
            self.trainer.monitors.put_summary(summary)


class MergeAllSummaries_RunWithOp(Callback):
    def __init__(self, period, key):
        self._period = period
        self._key = key

    def _setup_graph(self):
        self.summary_op = tf.summary.merge_all(self._key)
        if self.summary_op is not None:
            self._fetches = tf.train.SessionRunArgs(self.summary_op)
        else:
            self._fetches = None
        self._total = self.trainer.config.steps_per_epoch

    def _need_run(self):
        if self.local_step == self._total - 1:
            return True
        if self._period > 0 and (self.local_step + 1) % self._period == 0:
            return True
        return False

    def _before_run(self, ctx):
        if self._need_run():
            return self._fetches
        return None

    def _after_run(self, _, run_values):
        summary = run_values.results
        if summary is None:
            return
        self.trainer.monitors.put_summary(summary)


def MergeAllSummaries(period=0, run_alone=False, key=tf.GraphKeys.SUMMARIES):
    """
    Evaluate all summaries by `tf.summary.merge_all`, and write to logs.

    Args:
        period (int): by default the callback summarizes once every epoch.
            This option (if not set to 0) makes it additionally summarize every ``period`` steps.
        run_alone (bool): whether to evaluate the summaries alone.
            If True, summaries will be evaluated after each epoch alone.
            If False, summaries will be evaluated together with other
            `sess.run` calls, in the last step of each epoch.
            For :class:`SimpleTrainer`, it needs to be False because summary may
            depend on inputs.
        key (str): the collection of summary tensors. Same as in `tf.summary.merge_all`.

    Returns:
        a Callback.
    """
    period = int(period)
    if run_alone:
        return MergeAllSummaries_RunAlone(period, key)
    else:
        return MergeAllSummaries_RunWithOp(period, key)
