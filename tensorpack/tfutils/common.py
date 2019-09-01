# -*- coding: utf-8 -*-
# File: common.py

from collections import defaultdict
from six.moves import map
from tabulate import tabulate
import os
import sys
import psutil
import tensorflow as tf
import numpy as np

from ..compat import tfv1
from ..utils.argtools import graph_memoized
from ..utils.utils import find_library_full_path as find_library
from ..utils.nvml import NVMLContext
from ..libinfo import __git_version__

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


def collect_env_info():
    """
    Returns:
        str - a table contains important information about the environment
    """
    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("Tensorpack", __git_version__))
    data.append(("Numpy", np.__version__))

    data.append(("TensorFlow", tfv1.VERSION + "/" + tfv1.GIT_VERSION))
    data.append(("TF Compiler Version", tfv1.COMPILER_VERSION))
    has_cuda = tf.test.is_built_with_cuda()
    data.append(("TF CUDA support", has_cuda))

    try:
        from tensorflow.python.framework import test_util
        data.append(("TF MKL support", test_util.IsMklEnabled()))
    except Exception:
        pass

    try:
        from tensorflow.python.framework import test_util
        data.append(("TF XLA support", test_util.is_xla_enabled()))
    except Exception:
        pass

    if has_cuda:
        data.append(("Nvidia Driver", find_library("nvidia-ml")))
        data.append(("CUDA", find_library("cudart")))
        data.append(("CUDNN", find_library("cudnn")))
        data.append(("NCCL", find_library("nccl")))

        # List devices with NVML
        data.append(
            ("CUDA_VISIBLE_DEVICES",
             os.environ.get("CUDA_VISIBLE_DEVICES", str(None))))
        try:
            devs = defaultdict(list)
            with NVMLContext() as ctx:
                for idx, dev in enumerate(ctx.devices()):
                    devs[dev.name()].append(str(idx))

            for devname, devids in devs.items():
                data.append(
                    ("GPU " + ",".join(devids), devname))
        except Exception:
            data.append(("GPU", "Not found with NVML"))

    vram = psutil.virtual_memory()
    data.append(("Free RAM", "{:.2f}/{:.2f} GB".format(vram.available / 1024**3, vram.total / 1024**3)))
    data.append(("CPU Count", psutil.cpu_count()))

    # Other important dependencies:
    try:
        import horovod
        data.append(("Horovod", horovod.__version__))
    except ImportError:
        pass

    try:
        import cv2
        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass

    import msgpack
    data.append(("msgpack", ".".join([str(x) for x in msgpack.version])))

    has_prctl = True
    try:
        import prctl
        _ = prctl.set_pdeathsig  # noqa
    except Exception:
        has_prctl = False
    data.append(("python-prctl", has_prctl))

    return tabulate(data)


if __name__ == '__main__':
    print(collect_env_info())
