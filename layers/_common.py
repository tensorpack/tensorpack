#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: _common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from utils.summary import *

def layer_register(summary_activation=False):
    """
    summary_activation: default behavior of whether to summary the output of this layer
    """
    def wrapper(func):
        def inner(*args, **kwargs):
            name = args[0]
            assert isinstance(name, basestring)
            args = args[1:]
            do_summary = kwargs.pop(
                'summary_activation', summary_activation)

            with tf.variable_scope(name) as scope:
                ret = func(*args, **kwargs)
                if do_summary:
                    ndim = ret.get_shape().ndims
                    assert ndim >= 2, \
                        "Summary a scalar with histogram? Maybe use scalar instead. FIXME!"
                    add_activation_summary(ret, scope.name)
                return ret
        return inner
    return wrapper

def shape2d(a):
    """
    a: a int or tuple/list of length 2
    """
    if type(a) == int:
        return [a, a]
    if type(a) in [list, tuple]:
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))

def shape4d(a):
    # for use with tensorflow
    return [1] + shape2d(a) + [1]

