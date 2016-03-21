#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from ..utils.naming import *
import tensorflow as tf

def get_default_sess_config(mem_fraction=0.5):
    """
    Return a better config to use as default.
    Tensorflow default session config consume too much resources
    """
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction
    conf.gpu_options.allocator_type = 'BFC'
    conf.allow_soft_placement = True
    return conf

def get_global_step_var():
    """ get global_step variable in the current graph"""
    try:
        return tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    except KeyError:
        var = tf.Variable(
            0, trainable=False, name=GLOBAL_STEP_OP_NAME)
        return var

def get_global_step():
    """ get global_step value with current graph and session"""
    return tf.train.global_step(
        tf.get_default_session(),
        get_global_step_var())

