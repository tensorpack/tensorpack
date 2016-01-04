#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: validation_callback.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import itertools
from tqdm import tqdm

from ..utils import *
from ..utils.stat import *
from ..utils.summary import *
from .base import PeriodicCallback, Callback

__all__ = ['ValidationError', 'ValidationCallback']

class ValidationCallback(PeriodicCallback):
    running_graph = 'test'
    """
    Basic routine for validation callbacks.
    """
    def __init__(self, ds, prefix, period=1, cost_var_name='cost:0'):
        super(ValidationCallback, self).__init__(period)
        self.ds = ds
        self.prefix = prefix
        self.cost_var_name = cost_var_name

    def _before_train(self):
        self.input_vars = tf.get_collection(INPUT_VARS_KEY)
        self.cost_var = self.get_tensor(self.cost_var_name)
        self.writer = tf.get_collection(SUMMARY_WRITER_COLLECTION_KEY)[0]
        self._find_output_vars()

    def get_tensor(self, name):
        return self.graph.get_tensor_by_name(name)

    def _find_output_vars(self):
        pass

    def _get_output_vars(self):
        return []

    def _run_validation(self):
        """
        Generator to return inputs and outputs
        """
        cnt = 0
        cost_sum = 0

        output_vars = self._get_output_vars()
        output_vars.append(self.cost_var)
        with tqdm(total=self.ds.size()) as pbar:
            for dp in self.ds.get_data():
                feed = dict(itertools.izip(self.input_vars, dp))

                batch_size = dp[0].shape[0]   # assume batched input

                cnt += batch_size
                outputs = self.sess.run(output_vars, feed_dict=feed)
                cost = outputs[-1]
                # each batch might not have the same size in validation
                cost_sum += cost * batch_size
                yield (dp, outputs[:-1])
                pbar.update()

        cost_avg = cost_sum / cnt
        self.writer.add_summary(create_summary(
            '{}_cost'.format(self.prefix), cost_avg), self.global_step)
        logger.info("{}_cost: {:.4f}".format(self.prefix, cost_avg))

    def _trigger(self):
        for dp, outputs in self._run_validation():
            pass


class ValidationError(ValidationCallback):
    running_graph = 'test'
    """
    Validate the accuracy for the given wrong and cost variable
    Use under the following setup:
        wrong_var: integer, number of failed samples in this batch
        ds: batched dataset
    """
    def __init__(self, ds, prefix,
                 period=1,
                 wrong_var_name='wrong:0',
                 cost_var_name='cost:0'):
        super(ValidationError, self).__init__(
            ds, prefix, period, cost_var_name)
        self.wrong_var_name = wrong_var_name

    def _find_output_vars(self):
        self.wrong_var = self.get_tensor(self.wrong_var_name)

    def _get_output_vars(self):
        return [self.wrong_var]

    def _trigger(self):
        err_stat = Accuracy()
        for dp, outputs in self._run_validation():
            batch_size = dp[0].shape[0]   # assume batched input
            wrong = outputs[0]
            err_stat.feed(wrong, batch_size)

        self.writer.add_summary(create_summary(
            '{}_error'.format(self.prefix), err_stat.accuracy), self.global_step)
        logger.info("{}_error: {:.4f}".format(self.prefix, err_stat.accuracy))
