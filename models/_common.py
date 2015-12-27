#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: _common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from utils.summary import *
from utils import logger

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
            inputs = args[0]
            if isinstance(inputs, list):
                shape_str = ",".join(
                    map(str(x.get_shape().as_list()), inputs))
            else:
                shape_str = str(inputs.get_shape().as_list())
            logger.info("{} input: {}".format(name, shape_str))

            with tf.variable_scope(name) as scope:
                outputs = func(*args, **kwargs)
                if isinstance(outputs, list):
                    shape_str = ",".join(
                        map(str(x.get_shape().as_list()), outputs))
                    if do_summary:
                        for x in outputs:
                            add_activation_summary(x, scope.name)
                else:
                    shape_str = str(outputs.get_shape().as_list())
                    if do_summary:
                        add_activation_summary(outputs, scope.name)
                logger.info("{} output: {}".format(name, shape_str))
                return outputs
        return inner
    return wrapper

def shape2d(a):
    """
    a: a int or tuple/list of length 2
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))

def shape4d(a):
    # for use with tensorflow
    return [1] + shape2d(a) + [1]

