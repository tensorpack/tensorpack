#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import copy
from six.moves import zip
from contextlib import contextmanager
import operator

import tensorflow as tf

from ..tfutils.common import get_op_tensor_name
from ..utils import logger

__all__ = ['LeastLoadedDeviceSetter', 'OverrideToLocalVariable',
           'override_to_local_variable']


def get_tensors_inputs(placeholders, tensors, names):
    """
    Args:
        placeholders (list[Tensor]):
        tensors (list[Tensor]): list of tf.Tensor
        names (list[str]): names matching the tensors

    Returns:
        list[Tensor]: inputs to used with build_graph(),
            with the corresponding placeholders replaced by tensors.
    """
    assert len(tensors) == len(names), \
        "Input tensors {} and input names {} have different length!".format(
            tensors, names)
    ret = copy.copy(placeholders)
    placeholder_names = [p.name for p in placeholders]
    for name, tensor in zip(names, tensors):
        tensorname = get_op_tensor_name(name)[1]
        try:
            idx = placeholder_names.index(tensorname)
        except ValueError:
            logger.error("Name {} is not a model input!".format(tensorname))
            raise
        ret[idx] = tensor
    return ret


def get_sublist_by_names(lst, names):
    """
    Args:
        lst (list): list of objects with "name" property.

    Returns:
        list: a sublist of objects, matching names
    """
    orig_names = [p.name for p in lst]
    ret = []
    for name in names:
        try:
            idx = orig_names.index(name)
        except ValueError:
            logger.error("Name {} doesn't appear in lst {}!".format(
                name, str(orig_names)))
            raise
        ret.append(lst[idx])
    return ret


@contextmanager
def override_to_local_variable(enable=True):
    if enable:
        with tf.variable_scope(
                tf.get_variable_scope(),
                custom_getter=OverrideToLocalVariable()):
            yield
    else:
        yield


class OverrideToLocalVariable(object):
    """
    Ensures the created variable
    is in LOCAL_VARIABLES and not GLOBAL_VARIBLES collection.
    """
    def __call__(self, getter, name, *args, **kwargs):
        if 'collections' in kwargs:
            collections = kwargs['collections']
        if not collections:
            collections = set([tf.GraphKeys.GLOBAL_VARIABLES])
        else:
            collections = set(collections.copy())
        collections.remove(tf.GraphKeys.GLOBAL_VARIABLES)
        collections.add(tf.GraphKeys.LOCAL_VARIABLES)
        kwargs['collections'] = list(collections)
        return getter(name, *args, **kwargs)


# Copied from https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/variable_mgr.py
class LeastLoadedDeviceSetter(object):
    """ Helper class to assign variables on the least loaded ps-device."""
    def __init__(self, worker_device, ps_devices):
        """
        Args:
            worker_device: the device to use for compute ops.
            ps_devices: a list of device to use for Variable ops.
        """
        self.ps_devices = ps_devices
        self.worker_device = worker_device
        self.ps_sizes = [0] * len(self.ps_devices)

    def __call__(self, op):
        def sanitize_name(name):    # tensorflow/tensorflow#11484
            return tf.DeviceSpec.from_string(name).to_string()

        if op.device:
            return op.device
        if op.type not in ['Variable', 'VariableV2']:
            return sanitize_name(self.worker_device)

        device_index, _ = min(enumerate(
            self.ps_sizes), key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]
        var_size = op.outputs[0].get_shape().num_elements()
        self.ps_sizes[device_index] += var_size

        return sanitize_name(device_name)

    def __str__(self):
        return "LeastLoadedDeviceSetter-{}".format(self.worker_device)
