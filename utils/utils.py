#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

__all__ = ['create_summary', 'add_histogram_summary']

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

