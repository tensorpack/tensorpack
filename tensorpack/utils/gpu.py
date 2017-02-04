#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
from .utils import change_env

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
        int: the number of GPU from ``CUDA_VISIBLE_DEVICES``.
    """
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    assert env is not None, 'gpu not set!'  # TODO
    return len(env.split(','))
