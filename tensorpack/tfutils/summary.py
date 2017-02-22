# -*- coding: UTF-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import six
import tensorflow as tf
import re

from ..utils import logger
from ..utils.develop import log_deprecated
from ..utils.naming import MOVING_SUMMARY_OPS_KEY
from .tower import get_current_tower_context
from .symbolic_functions import rms
from .common import get_global_step_var

__all__ = ['create_scalar_summary', 'add_param_summary', 'add_activation_summary',
           'add_moving_summary']


def create_scalar_summary(name, v):
    """
    Returns:
        tf.Summary: a tf.Summary object with name and simple scalar value v.
    """
    assert isinstance(name, six.string_types), type(name)
    v = float(v)
    s = tf.Summary()
    s.value.add(tag=name, simple_value=v)
    return s


def add_activation_summary(x, name=None):
    """
    Add summary for an activation tensor x.  If name is None, use x.name.

    Args:
        x (tf.Tensor): the tensor to summary.
    """
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    ndim = x.get_shape().ndims
    if ndim < 2:
        logger.warn("Cannot summarize scalar activation {}".format(x.name))
        return
    if name is None:
        name = x.name
    with tf.name_scope('activation-summary'):
        tf.summary.histogram(name, x)
        tf.summary.scalar(name + '-sparsity', tf.nn.zero_fraction(x))
        tf.summary.scalar(name + '-rms', rms(x))


def add_param_summary(*summary_lists):
    """
    Add summary Ops for all trainable variables matching the regex.

    Args:
        summary_lists (list): each is (regex, [list of summary type to perform]).
        Summary type can be 'mean', 'scalar', 'histogram', 'sparsity', 'rms'
    """
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    if len(summary_lists) == 1 and isinstance(summary_lists[0], list):
        log_deprecated(text="Use positional args to call add_param_summary() instead of a list.")
        summary_lists = summary_lists[0]

    def perform(var, action):
        ndim = var.get_shape().ndims
        name = var.name.replace(':0', '')
        if action == 'scalar':
            assert ndim == 0, "Scalar summary on high-dimension data. Maybe you want 'mean'?"
            tf.summary.scalar(name, var)
            return
        assert ndim > 0, "Cannot perform {} summary on scalar data".format(action)
        if action == 'histogram':
            tf.summary.histogram(name, var)
            return
        if action == 'sparsity':
            tf.summary.scalar(name + '-sparsity', tf.nn.zero_fraction(var))
            return
        if action == 'mean':
            tf.summary.scalar(name + '-mean', tf.reduce_mean(var))
            return
        if action == 'rms':
            tf.summary.scalar(name + '-rms', rms(var))
            return
        raise RuntimeError("Unknown summary type: {}".format(action))

    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    with tf.name_scope('param-summary'):
        for p in params:
            name = p.name
            for rgx, actions in summary_lists:
                if not rgx.endswith('$'):
                    rgx = rgx + '(:0)?$'
                if re.match(rgx, name):
                    for act in actions:
                        perform(p, act)


def add_moving_summary(v, *args, **kwargs):
    """
    Args:
        v (tf.Tensor or list): tensor or list of tensors to summary. Must have
            scalar type.
        args: tensors to summary (support positional arguments)
        decay (float): the decay rate. Defaults to 0.95.
        collection (str): the name of the collection to add EMA-maintaining ops.
            The default will work together with the default
            :class:`MovingAverageSummary` callback.
    """
    decay = kwargs.pop('decay', 0.95)
    coll = kwargs.pop('collection', MOVING_SUMMARY_OPS_KEY)
    assert len(kwargs) == 0, "Unknown arguments: " + str(kwargs)

    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    if not isinstance(v, list):
        v = [v]
    v.extend(args)
    for x in v:
        assert isinstance(x, tf.Tensor), x
        assert x.get_shape().ndims == 0, x.get_shape()
    # TODO will produce tower0/xxx?
    with tf.name_scope(None):
        averager = tf.train.ExponentialMovingAverage(
            decay, num_updates=get_global_step_var(), name='EMA')
        avg_maintain_op = averager.apply(v)
    for c in v:
        # TODO do this in the EMA callback?
        name = re.sub('tower[p0-9]+/', '', c.op.name)
        tf.summary.scalar(name + '-summary', averager.average(c))

    tf.add_to_collection(coll, avg_maintain_op)
