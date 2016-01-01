#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import logger
from naming import *

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
    tf.histogram_summary(name + '/activations', x)
    tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(x))
    # TODO avoid repeating activations on multiple GPUs

def add_histogram_summary(regex):
    """
    Add histogram summary for all trainable variables matching the regex
    """
    import re
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for p in params:
        name = p.name
        if re.search(regex, name):
            tf.histogram_summary(name, p)

def summary_moving_average(cost_var):
    """ Create a MovingAverage op and summary for all variables in
        COST_VARS_KEY, SUMMARY_VARS_KEY, as well as the argument
        Return a op to maintain these average
    """
    global_step_var = tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    averager = tf.train.ExponentialMovingAverage(
        0.9, num_updates=global_step_var, name='moving_averages')
    vars_to_summary = [cost_var] + \
            tf.get_collection(SUMMARY_VARS_KEY) + \
            tf.get_collection(COST_VARS_KEY)
    avg_maintain_op = averager.apply(vars_to_summary)
    for c in vars_to_summary:
        tf.scalar_summary(c.op.name, averager.average(c))
    return avg_maintain_op

