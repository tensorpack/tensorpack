#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

__all__ = ['create_summary']

def create_summary(name, v):
# TODO support image or histogram
    """
    Args: v: a value

    """
    assert isinstance(name, basestring), type(name)
    v = float(v)
    s = tf.Summary()
    s.value.add(tag=name, simple_value=v)
    return s

