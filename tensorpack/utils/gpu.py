# -*- coding: utf-8 -*-
# File: gpu.py


import os

from . import logger
from .concurrency import subproc_call
from .nvml import NVMLContext
from .utils import change_env

__all__ = ['change_gpu', 'get_nr_gpu', 'get_num_gpu']


def change_gpu(val):
    """
    Args:
        val: an integer, the index of the GPU or -1 to disable GPU.

    Returns:
        a context where ``CUDA_VISIBLE_DEVICES=val``.
    """
    val = str(val)
    if val == '-1':
        val = ''
    return change_env('CUDA_VISIBLE_DEVICES', val)


def get_num_gpu():
    """
    Returns:
        int: #available GPUs in CUDA_VISIBLE_DEVICES, or in the system.
    """

    def warn_return(ret, message):
        try:
            import tensorflow as tf
        except ImportError:
            return ret

        built_with_cuda = tf.test.is_built_with_cuda()
        if not built_with_cuda and ret > 0:
            logger.warn(message + "But TensorFlow was not built with CUDA support and could not use GPUs!")
        return ret

    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env:
        return warn_return(len(env.split(',')), "Found non-empty CUDA_VISIBLE_DEVICES. ")
    output, code = subproc_call("nvidia-smi -L", timeout=5)
    if code == 0:
        output = output.decode('utf-8')
        return warn_return(len(output.strip().split('\n')), "Found nvidia-smi. ")
    try:
        # Use NVML to query device properties
        with NVMLContext() as ctx:
            return warn_return(ctx.num_devices(), "NVML found nvidia devices. ")
    except Exception:
        # Fallback
        logger.info("Loading local devices by TensorFlow ...")

        try:
            import tensorflow as tf
            # available since TF 1.14
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        except AttributeError:
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            # Note this will initialize all GPUs and therefore has side effect
            # https://github.com/tensorflow/tensorflow/issues/8136
            gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
        return len(gpu_devices)


get_nr_gpu = get_num_gpu
