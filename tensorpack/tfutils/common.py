#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from six.moves import map

from ..utils.naming import (
    GLOBAL_STEP_VAR_NAME,
    GLOBAL_STEP_OP_NAME)
from ..utils.argtools import memoized

__all__ = ['get_default_sess_config',

           'get_global_step_value',
           'get_global_step_var',
           #'get_local_step_var',

           'get_op_tensor_name',
           'get_tensors_by_names',
           'get_op_or_tensor_by_name',
           'get_name_scope_name',
           ]


def get_default_sess_config(mem_fraction=0.99):
    """
    Return a better session config to use as default.
    Tensorflow default session config consume too much resources.

    Args:
        mem_fraction(float): fraction of memory to use.
    Returns:
        tf.ConfigProto: the config to use.
    """
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction
    conf.gpu_options.allocator_type = 'BFC'
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
    # conf.log_device_placement = True
    return conf


@memoized
def get_global_step_var():
    """
    Returns:
        tf.Tensor: the global_step variable in the current graph. create if
        doesn't exist.
    """
    try:
        return tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    except KeyError:
        scope = tf.get_variable_scope()
        assert scope.name == '', \
            "The global_step variable should be created under the root variable scope!"
        with tf.variable_scope(scope, reuse=False), \
                tf.name_scope(None):
            var = tf.get_variable(GLOBAL_STEP_OP_NAME,
                                  initializer=tf.constant(0, dtype=tf.int64),
                                  trainable=False, dtype=tf.int64)
        return var


def get_global_step_value():
    """
    Returns:
        int: global_step value in current graph and session"""
    return tf.train.global_step(
        tf.get_default_session(),
        get_global_step_var())


# @memoized
# def get_local_step_var():
#     try:
#         return tf.get_default_graph().get_tensor_by_name(LOCAL_STEP_VAR_NAME)
#     except KeyError:
#         logger.warn("get_local_step_var() is only available to use in callbacks!")
#         raise


def get_op_tensor_name(name):
    """
    Will automatically determine if ``name`` is a tensor name (ends with ':x')
    or a op name.
    If it is an op name, the corresponding tensor name is assumed to be ``op_name + ':0'``.

    Args:
        name(str): name of an op or a tensor
    Returns:
        tuple: (op_name, tensor_name)
    """
    if len(name) >= 3 and name[-2] == ':':
        return name[:-2], name
    else:
        return name, name + ':0'


def get_tensors_by_names(names):
    """
    Get a list of tensors in the default graph by a list of names.

    Args:
        names (list):
    """
    ret = []
    G = tf.get_default_graph()
    for n in names:
        opn, varn = get_op_tensor_name(n)
        ret.append(G.get_tensor_by_name(varn))
    return ret


def get_op_or_tensor_by_name(name):
    """
    Get either tf.Operation of tf.Tensor from names.

    Args:
        name (list[str] or str): names of operations or tensors.
    """
    G = tf.get_default_graph()

    def f(n):
        if len(n) >= 3 and n[-2] == ':':
            return G.get_tensor_by_name(n)
        else:
            return G.get_operation_by_name(n)

    if not isinstance(name, list):
        return f(name)
    else:
        return list(map(f, name))


def get_name_scope_name():
    """
    Returns:
        str: the name of the current name scope, without the ending '/'.
    """
    g = tf.get_default_graph()
    s = "RANDOM_STR_ABCDEFG"
    unique = g.unique_name(s)
    scope = unique[:-len(s)].rstrip('/')
    return scope
