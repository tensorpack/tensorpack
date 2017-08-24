#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from six.moves import map
from ..utils.argtools import graph_memoized

__all__ = ['get_default_sess_config',
           'get_global_step_value',
           'get_global_step_var',
           # 'get_op_tensor_name',
           # 'get_tensors_by_names',
           # 'get_op_or_tensor_by_name',
           # 'get_tf_version_number',
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

    conf.allow_soft_placement = True
    # conf.log_device_placement = True

    # https://github.com/tensorflow/tensorflow/issues/9322#issuecomment-295758107
    # can speed up a bit
    conf.intra_op_parallelism_threads = 1
    conf.inter_op_parallelism_threads = 0

    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction
    if get_tf_version_number() >= 1.2:
        conf.gpu_options.force_gpu_compatible = True

    conf.gpu_options.allocator_type = 'BFC'
    conf.gpu_options.allow_growth = True

    # May hurt performance
    # conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    return conf


@graph_memoized
def get_global_step_var():
    """
    Returns:
        tf.Tensor: the global_step variable in the current graph. create if
        doesn't exist.
    """
    scope = tf.get_variable_scope()
    assert scope.name == '', \
        "The global_step variable should be created under the root variable scope!"
    assert not scope.reuse, \
        "The global_step variable shouldn't be called under a reuse variable scope!"
    if get_tf_version_number() <= 1.0:
        var = tf.get_variable('global_step',
                              initializer=tf.constant(0, dtype=tf.int64),
                              trainable=False, dtype=tf.int64)
        tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, var)
    else:
        var = tf.train.get_or_create_global_step()
    return var


def get_global_step_value():
    """
    Returns:
        int: global_step value in current graph and session"""
    return tf.train.global_step(
        tf.get_default_session(),
        get_global_step_var())


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


def get_tf_version_number():
    """
    Return a float (for comparison), indicating tensorflow version.
    """
    return float('.'.join(tf.VERSION.split('.')[:2]))
