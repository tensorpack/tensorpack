# -*- coding: utf-8 -*-
# File: gpu.py


import os
from .utils import change_env
from . import logger
from .nvml import NVMLContext
from .concurrency import subproc_call

__all__ = ['change_gpu', 'get_nr_gpu', 'get_num_gpu']


def change_gpu(val):
    """
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
            logger.warn(message + "But TensorFlow was not built with CUDA support!")
        return ret

    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env is not None:
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
        # Note this will initialize all GPUs and therefore has side effect
        # https://github.com/tensorflow/tensorflow/issues/8136
        logger.info("Loading local devices by TensorFlow ...")
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


get_nr_gpu = get_num_gpu
