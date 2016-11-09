# -*- coding: UTF-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import six
import tensorflow as tf
import re

from ..utils import *
from .tower import get_current_tower_context
from . import get_global_step_var
from .symbolic_functions import rms

__all__ = ['create_summary', 'add_param_summary', 'add_activation_summary',
           'add_moving_summary', 'summary_moving_average']

def create_summary(name, v):
    """
    Return a tf.Summary object with name and simple scalar value v
    """
    assert isinstance(name, six.string_types), type(name)
    v = float(v)
    s = tf.Summary()
    s.value.add(tag=name, simple_value=v)
    return s

def add_activation_summary(x, name=None):
    """
    Add summary to graph for an activation tensor x.
    If name is None, use x.name.
    """
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    ndim = x.get_shape().ndims
    assert ndim >= 2, \
        "Summary a scalar with histogram? Maybe use scalar instead. FIXME!"
    if name is None:
        name = x.name
    with tf.name_scope('act_summary'):
        tf.histogram_summary(name + '/activation', x)
        tf.scalar_summary(name + '/activation_sparsity', tf.nn.zero_fraction(x))
        tf.scalar_summary(
                name + '/activation_rms', rms(x))

def add_param_summary(summary_lists):
    """
    Add summary for all trainable variables matching the regex

    :param summary_lists: list of (regex, [list of summary type to perform]).
        Type can be 'mean', 'scalar', 'histogram', 'sparsity', 'rms'
    """
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    def perform(var, action):
        ndim = var.get_shape().ndims
        name = var.name.replace(':0', '')
        if action == 'scalar':
            assert ndim == 0, "Scalar summary on high-dimension data. Maybe you want 'mean'?"
            tf.scalar_summary(name, var)
            return
        assert ndim > 0, "Cannot perform {} summary on scalar data".format(action)
        if action == 'histogram':
            tf.histogram_summary(name, var)
            return
        if action == 'sparsity':
            tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(var))
            return
        if action == 'mean':
            tf.scalar_summary(name + '/mean', tf.reduce_mean(var))
            return
        if action == 'rms':
            tf.scalar_summary(name + '/rms', rms(var))
            return
        raise RuntimeError("Unknown summary type: {}".format(action))

    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    with tf.name_scope('param_summary'):
        for p in params:
            name = p.name
            for rgx, actions in summary_lists:
                if not rgx.endswith('$'):
                    rgx = rgx + '(:0)?$'
                if re.match(rgx, name):
                    for act in actions:
                        perform(p, act)

def add_moving_summary(v, *args):
    """
    :param v: tensor or list of tensor to summary
    :param args: tensors to summary
    """
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    if not isinstance(v, list):
        v = [v]
    v.extend(args)
    for x in v:
        assert x.get_shape().ndims == 0
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, x)

def summary_moving_average(tensors=None):
    """
    Create a MovingAverage op and summary for tensors
    :param tensors: list of tf.Tensor to summary. default to the collection MOVING_SUMMARY_VARS_KEY
    :returns: a op to maintain these average.
    """
    if tensors is None:
        tensors = tf.get_collection(MOVING_SUMMARY_VARS_KEY)
    with tf.name_scope('EMA_summary'):
        # TODO will produce EMA_summary/tower0/xxx. not elegant
        with tf.name_scope(None):
            averager = tf.train.ExponentialMovingAverage(
                0.99, num_updates=get_global_step_var(), name='EMA')
        avg_maintain_op = averager.apply(tensors)
        for idx, c in enumerate(tensors):
            name = re.sub('tower[p0-9]+/', '', c.op.name)
            tf.scalar_summary(name, averager.average(c))
        return avg_maintain_op

