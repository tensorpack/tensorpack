# -*- coding: utf-8 -*-
# File: common.py

import tensorflow as tf

from ..compat import tfv1
from ..utils.argtools import graph_memoized
from .collect_env import collect_env_info


__all__ = ['get_default_sess_config',
           'get_global_step_value',
           'get_global_step_var',
           'get_tf_version_tuple',
           'collect_env_info'
           # 'get_op_tensor_name',
           # 'get_tensors_by_names',
           # 'get_op_or_tensor_by_name',
           ]


def get_default_sess_config(mem_fraction=0.99):
    """
    Return a tf.ConfigProto to use as default session config.
    You can modify the returned config to fit your needs.

    Args:
        mem_fraction(float): see the `per_process_gpu_memory_fraction` option
            in TensorFlow's GPUOptions protobuf:
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto

    Returns:
        tf.ConfigProto: the config to use.
    """
    conf = tfv1.ConfigProto()

    conf.allow_soft_placement = True
    # conf.log_device_placement = True

    conf.intra_op_parallelism_threads = 1
    conf.inter_op_parallelism_threads = 0
    # TF benchmark use cpu_count() - gpu_thread_count(), e.g. 80 - 8 * 2
    # Didn't see much difference.

    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction

    # This hurt performance of large data pipeline:
    # https://github.com/tensorflow/benchmarks/commit/1528c46499cdcff669b5d7c006b7b971884ad0e6
    # conf.gpu_options.force_gpu_compatible = True

    conf.gpu_options.allow_growth = True

    # from tensorflow.core.protobuf import rewriter_config_pb2 as rwc
    # conf.graph_options.rewrite_options.memory_optimization = \
    #     rwc.RewriterConfig.HEURISTICS

    # May hurt performance?
    # conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # conf.graph_options.place_pruned_graph = True
    return conf


@graph_memoized
def get_global_step_var():
    """
    Returns:
        tf.Tensor: the global_step variable in the current graph. Create if doesn't exist.
    """
    scope = tfv1.VariableScope(reuse=False, name='')  # the root vs
    with tfv1.variable_scope(scope):
        var = tfv1.train.get_or_create_global_step()
    return var


def get_global_step_value():
    """
    Returns:
        int: global_step value in current graph and session

    Has to be called under a default session.
    """

    return tfv1.train.global_step(
        tfv1.get_default_session(),
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
    G = tfv1.get_default_graph()
    for n in names:
        opn, varn = get_op_tensor_name(n)
        ret.append(G.get_tensor_by_name(varn))
    return ret


def get_op_or_tensor_by_name(name):
    """
    Get either tf.Operation of tf.Tensor from names.

    Args:
        name (list[str] or str): names of operations or tensors.

    Raises:
        KeyError, if the name doesn't exist
    """
    G = tfv1.get_default_graph()

    def f(n):
        if len(n) >= 3 and n[-2] == ':':
            return G.get_tensor_by_name(n)
        else:
            return G.get_operation_by_name(n)

    if not isinstance(name, list):
        return f(name)
    else:
        return list(map(f, name))


def gpu_available_in_session():
    sess = tfv1.get_default_session()
    for dev in sess.list_devices():
        if dev.device_type.lower() == 'gpu':
            return True
    return False


def get_tf_version_tuple():
    """
    Return TensorFlow version as a 2-element tuple (for comparison).
    """
    return tuple(map(int, tf.__version__.split('.')[:2]))
