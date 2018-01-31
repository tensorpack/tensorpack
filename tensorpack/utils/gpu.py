#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gpu.py


import os
from .utils import change_env
from .concurrency import subproc_call

__all__ = ['change_gpu', 'get_nr_gpu']


def change_gpu(val):
    """
    Returns:
        a context where ``CUDA_VISIBLE_DEVICES=val``.
    """
    val = str(val)
    if val == '-1':
        val = ''
    return change_env('CUDA_VISIBLE_DEVICES', val)


def get_nr_gpu():
    """
    Returns:
        int: #available GPUs in CUDA_VISIBLE_DEVICES, or in the system.
    """
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env is not None:
        return len(env.split(','))
    output, code = subproc_call("nvidia-smi -L", timeout=5)
    if code != 0:
        return 0
    output = output.decode('utf-8')
    return len(output.strip().split('\n'))
