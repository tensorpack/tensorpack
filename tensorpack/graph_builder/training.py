#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: training.py

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import six

from ..tfutils.gradproc import FilterNoneGrad
from ..tfutils.tower import TowerContext


@six.add_metaclass(ABCMeta)
class GraphBuilder(object):
    @abstractmethod
    def build(*args, **kwargs):
        pass


class SimpleGraphBuilder(GraphBuilder):
    """
    Build the graph for single-cost single-optimizer single-tower training.
    """
    def build(self, input, get_cost_fn, get_opt_fn):
        """
        Args:
            input (InputSource): should have been setup already
            get_cost_fn ([tf.Tensor] -> tf.Tensor): a callable,
                taking several tensors as input and returns a cost tensor.
            get_opt_fn (None -> tf.train.Optimizer): a callable that returns an optimizer

        Returns:
            tf.Operation: the training op
        """
        with TowerContext('', is_training=True) as ctx:
            cost = get_cost_fn(*input.get_input_tensors())

            varlist = ctx.filter_vars_by_vs_name(tf.trainable_variables())
            opt = get_opt_fn()
            grads = opt.compute_gradients(
                cost, var_list=varlist,
                gate_gradients=False, colocate_gradients_with_ops=True)
            grads = FilterNoneGrad().process(grads)
            train_op = opt.apply_gradients(grads, name='min_op')
            return train_op
