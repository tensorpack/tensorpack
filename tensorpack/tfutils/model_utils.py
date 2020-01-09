# -*- coding: utf-8 -*-
# File: model_utils.py
# Author: tensorpack contributors

from ..compat import tfv1 as tf
from tabulate import tabulate
from termcolor import colored

from .common import get_op_tensor_name
from ..utils import logger

__all__ = []


def describe_trainable_vars():
    """
    Print a description of the current model parameters.
    Skip variables starting with "tower", as they are just duplicates built by data-parallel logic.
    """
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if len(train_vars) == 0:
        logger.warn("No trainable variables in the graph!")
        return
    total = 0
    total_bytes = 0
    data = []
    for v in train_vars:
        if v.name.startswith('tower'):
            continue
        shape = v.get_shape()
        ele = shape.num_elements()
        if ele is None:
            logger.warn("Shape of variable {} is not fully defined but {}.".format(v.name, shape))
            ele = 0
        try:
            shape = shape.as_list()
        except ValueError:
            shape = '<unknown>'

        total += ele
        total_bytes += ele * v.dtype.size
        data.append([get_op_tensor_name(v.name)[0], shape, ele, v.device, v.dtype.base_dtype.name])
    headers = ['name', 'shape', '#elements', 'device', 'dtype']

    dtypes = list({x[4] for x in data})
    if len(dtypes) == 1 and dtypes[0] == "float32":
        # don't log the dtype if all vars are float32 (default dtype)
        for x in data:
            del x[4]
        del headers[4]

    devices = {x[3] for x in data}
    if len(devices) == 1:
        # don't log the device if all vars on the same device
        for x in data:
            del x[3]
        del headers[3]

    table = tabulate(data, headers=headers)

    size_mb = total_bytes / 1024.0**2
    summary_msg = colored(
        "\nNumber of trainable variables: {}".format(len(data)) +
        "\nNumber of parameters (elements): {}".format(total) +
        "\nStorage space needed for all trainable variables: {:.02f}MB".format(size_mb),
        'cyan')
    logger.info(colored("List of Trainable Variables: \n", 'cyan') + table + summary_msg)


def get_shape_str(tensors):
    """
    Internally used by layer registry, to print shapes of inputs/outputs of layers.

    Args:
        tensors (list or tf.Tensor): a tensor or a list of tensors
    Returns:
        str: a string to describe the shape
    """
    if isinstance(tensors, (list, tuple)):
        for v in tensors:
            assert isinstance(v, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(v))
        shape_str = ", ".join(map(get_shape_str, tensors))
    else:
        assert isinstance(tensors, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(tensors))
        shape_str = str(tensors.get_shape().as_list()).replace("None", "?")
    return shape_str
