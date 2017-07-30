# -*- coding: UTF-8 -*-
# File: model_utils.py
# Author: tensorpack contributors

import tensorflow as tf
from termcolor import colored
from tabulate import tabulate

from ..utils import logger

__all__ = []


def describe_trainable_vars():
    """
    Print a description of the current model parameters.
    Skip variables starting with "tower".
    """
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if len(train_vars) == 0:
        logger.warn("No trainable variables in the graph!")
        return
    total = 0
    data = []
    devices = set()
    for v in train_vars:
        if v.name.startswith('tower'):
            continue
        shape = v.get_shape()
        ele = shape.num_elements()
        total += ele
        devices.add(v.device)
        data.append([v.name, shape.as_list(), ele, v.device])

    if len(devices) == 1:
        # don't log the device if all vars on the same device
        for d in data:
            d.pop()
        table = tabulate(data, headers=['name', 'shape', 'dim'])
    else:
        table = tabulate(data, headers=['name', 'shape', 'dim', 'device'])

    size_mb = total * 4 / 1024.0**2
    summary_msg = colored(
        "\nTotal #vars={}, #param={} ({:.02f} MB assuming all float32)".format(
            len(data), total, size_mb), 'cyan')
    logger.info(colored("Model Parameters: \n", 'cyan') + table + summary_msg)


def get_shape_str(tensors):
    """
    Args:
        tensors (list or tf.Tensor): a tensor or a list of tensors
    Returns:
        str: a string to describe the shape
    """
    if isinstance(tensors, (list, tuple)):
        for v in tensors:
            assert isinstance(v, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(v))
        shape_str = ",".join(
            map(lambda x: str(x.get_shape().as_list()), tensors))
    else:
        assert isinstance(tensors, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(tensors))
        shape_str = str(tensors.get_shape().as_list())
    return shape_str
