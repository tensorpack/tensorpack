#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from ..utils.naming import *
import tensorflow as tf
from copy import copy
import six
from contextlib import contextmanager

__all__ = ['get_default_sess_config',
           'get_global_step',
           'get_global_step_var',
           'get_op_var_name',
           'get_vars_by_names',
           'backup_collection',
           'restore_collection',
           'clear_collection',
           'freeze_collection']

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
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
    #conf.log_device_placement = True
    return conf

def get_global_step_var():
    """ :returns: the global_step variable in the current graph. create if not existed"""
    try:
        return tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    except KeyError:
        scope = tf.get_variable_scope()
        assert scope.name == '', \
                "Creating global_step_var under a variable scope would cause problems!"
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

def get_vars_by_names(names):
    """
    Get a list of variables in the default graph by a list of names
    """
    ret = []
    G = tf.get_default_graph()
    for n in names:
        opn, varn = get_op_var_name(n)
        ret.append(G.get_tensor_by_name(varn))
    return ret

def backup_collection(keys):
    ret = {}
    for k in keys:
        ret[k] = copy(tf.get_collection(k))
    return ret

def restore_collection(backup):
    for k, v in six.iteritems(backup):
        del tf.get_collection_ref(k)[:]
        tf.get_collection_ref(k).extend(v)

def clear_collection(keys):
    for k in keys:
        del tf.get_collection_ref(k)[:]

@contextmanager
def freeze_collection(keys):
    backup = backup_collection(keys)
    yield
    restore_collection(backup)
