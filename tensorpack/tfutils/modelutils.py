# -*- coding: UTF-8 -*-
# File: modelutils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from termcolor import colored

from ..utils import logger

__all__ = ['describe_model', 'get_shape_str']


def describe_model():
    """ Print a description of the current model parameters """
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if len(train_vars) == 0:
        logger.info("No trainable variables in the graph!")
        return
    msg = [""]
    total = 0
    for v in train_vars:
        shape = v.get_shape()
        ele = shape.num_elements()
        total += ele
        msg.append("{}: shape={}, dim={}".format(
            v.name, shape.as_list(), ele))
    size_mb = total * 4 / 1024.0**2
    msg.append(colored(
        "Total #param={} ({:.02f} MB assuming all float32)".format(total, size_mb), 'cyan'))
    logger.info(colored("Model Parameters: ", 'cyan') + '\n'.join(msg))


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
