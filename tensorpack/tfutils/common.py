#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from ..utils.naming import *
import tensorflow as tf

def get_default_sess_config(mem_fraction=0.9):
    """
    Return a better session config to use as default.
    Tensorflow default session config consume too much resources.

    :param mem_fraction: fraction of memory to use.
    :returns: a `tf.ConfigProto` object.
    """
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction
    conf.gpu_options.allocator_type = 'BFC'
    conf.allow_soft_placement = True
    #conf.log_device_placement = True
    return conf

def get_global_step_var():
    """ :returns: the global_step variable in the current graph. create if not existed"""
    try:
        return tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    except KeyError:
        var = tf.Variable(
            0, trainable=False, name=GLOBAL_STEP_OP_NAME)
        return var

def get_global_step():
    """ :returns: global_step value in current graph and session"""
    return tf.train.global_step(
        tf.get_default_session(),
        get_global_step_var())

def get_op_var_name(name):
    """
    Variable name is assumed to be ``op_name + ':0'``

    :param name: an op or a variable name
    :returns: (op_name, variable_name)
    """
    if name.endswith(':0'):
        return name[:-2], name
    else:
        return name, name + ':0'
