#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from . import logger, get_global_step_var
from .naming import *

def create_summary(name, v):
    """
    Return a tf.Summary object with name and simple value v
    Args: v: a value

    """
    assert isinstance(name, basestring), type(name)
    v = float(v)
    s = tf.Summary()
    s.value.add(tag=name, simple_value=v)
    return s

def add_activation_summary(x, name=None):
    """
    Summary for an activation tensor x.
    If name is None, use x.name
    """
    ndim = x.get_shape().ndims
    assert ndim >= 2, \
        "Summary a scalar with histogram? Maybe use scalar instead. FIXME!"
    if name is None:
        name = x.name
    tf.histogram_summary(name + '/activation', x)
    tf.scalar_summary(name + '/activation_sparsity', tf.nn.zero_fraction(x))

def add_param_summary(regex):
    """
    Add summary for all trainable variables matching the regex
    """
    import re
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for p in params:
        name = p.name
        if re.search(regex, name):
            if p.get_shape().ndims == 0:
                tf.scalar_summary(name, p)
            else:
                #tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(p))
                tf.histogram_summary(name, p)

def summary_moving_average(cost_var):
    """ Create a MovingAverage op and summary for all variables in
        MOVING_SUMMARY_VARS_KEY, as well as the argument
        Return a op to maintain these average
    """
    global_step_var = get_global_step_var()
    averager = tf.train.ExponentialMovingAverage(
        0.99, num_updates=global_step_var, name='moving_averages')
    vars_to_summary = [cost_var] + \
            tf.get_collection(MOVING_SUMMARY_VARS_KEY)
    avg_maintain_op = averager.apply(vars_to_summary)
    for idx, c in enumerate(vars_to_summary):
        name = c.op.name
        if idx == 0:
            name = 'train_cost'
        tf.scalar_summary(name, averager.average(c))
    return avg_maintain_op

