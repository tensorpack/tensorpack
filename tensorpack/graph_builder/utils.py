#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py


from contextlib import contextmanager
import operator
import tensorflow as tf

from ..tfutils.common import get_tf_version_number


__all__ = ['LeastLoadedDeviceSetter',
           'OverrideCachingDevice',
           'override_to_local_variable',
           'allreduce_grads',
           'average_grads']


"""
Some utilities for building the graph.
"""


def _replace_global_by_local(kwargs):
    if 'collections' in kwargs:
        collections = kwargs['collections']
    if not collections:
        collections = set([tf.GraphKeys.GLOBAL_VARIABLES])
    else:
        collections = set(collections.copy())
    collections.remove(tf.GraphKeys.GLOBAL_VARIABLES)
    collections.add(tf.GraphKeys.LOCAL_VARIABLES)
    kwargs['collections'] = list(collections)


@contextmanager
def override_to_local_variable(enable=True):
    if enable:

        def custom_getter(getter, name, *args, **kwargs):
            _replace_global_by_local(kwargs)
            return getter(name, *args, **kwargs)

        orig_vs = tf.get_variable_scope()
        if get_tf_version_number() >= 1.5:
            with tf.variable_scope(
                    orig_vs,
                    custom_getter=custom_getter,
                    auxiliary_name_scope=False):
                yield
        else:
            if get_tf_version_number() >= 1.2:
                ns = tf.get_default_graph().get_name_scope()
            else:
                ns = orig_vs.original_name_scope
            with tf.variable_scope(
                    orig_vs, custom_getter=custom_getter):
                with tf.name_scope(ns + '/'):
                    yield
    else:
        yield


# https://github.com/tensorflow/benchmarks/blob/48cbef14a592e02a14beee8e9aef3ad22cadaed1/scripts/tf_cnn_benchmarks/variable_mgr_util.py#L192-L218
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


def allreduce_grads(all_grads):
    """
    All-reduce average the gradients among devices. Results are broadcasted to all devices.

    Args:
        all_grads (K x N x 2): A list of K lists. Each of the list is a list of N (grad, var) tuples.
            The variables have to be the same across the K lists.

    Returns:
        (K x N x 2): same as input, but each grad is replaced by the average over K lists.
    """
    from tensorflow.contrib import nccl
    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads
    new_all_grads = []  # NVar * NGPU * 2
    with tf.name_scope('AvgGrad'):
        for grad_and_vars in zip(*all_grads):
            v = grad_and_vars[0][1]
            grads = [g for g, _ in grad_and_vars]
            summed = nccl.all_sum(grads)

            grads_for_a_var = []
            for (_, v), g in zip(grad_and_vars, summed):
                with tf.device(g.device):
                    # tensorflow/benchmarks didn't average gradients
                    g = tf.multiply(g, 1.0 / nr_tower)
                    grads_for_a_var.append((g, v))
            new_all_grads.append(grads_for_a_var)

    # transpose
    ret = [k for k in zip(*new_all_grads)]
    return ret


def average_grads(all_grads, colocation=True, devices=None):
    """
    Average the gradients.

    Args:
        all_grads (K x N x 2): A list of K lists. Each of the list is a list of N (grad, var) tuples.
            The variables have to be the same across the K lists.
        colocation (bool): colocate gradient averaging on the device of the variable.
        devices (list[str]): assign the averaging to these device in
            round-robin. Cannot be used together with ``colocation``.

    Returns:
        (N x 2): A list of N (grad, var) tuples, where grad is averaged over K.
    """
    assert not (devices is not None and colocation)
    if devices is not None:
        assert isinstance(devices, list), devices

    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads[0]
    ret = []
    with tf.name_scope('AvgGrad'):
        for idx, grad_and_vars in enumerate(zip(*all_grads)):
            # Ngpu * 2
            v = grad_and_vars[0][1]
            grads = [g for (g, _) in grad_and_vars]

            if colocation:
                with tf.device(v.device):       # colocate summed grad with var
                    grad = tf.multiply(
                        tf.add_n(grads), 1.0 / nr_tower)
            elif devices is None:
                grad = tf.multiply(
                    tf.add_n(grads), 1.0 / nr_tower)
            else:
                dev = devices[idx % len(devices)]
                with tf.device(dev):
                    grad = tf.multiply(
                        tf.add_n(grads), 1.0 / nr_tower)
            ret.append((grad, v))
    return ret


# https://github.com/tensorflow/benchmarks/blob/48cbef14a592e02a14beee8e9aef3ad22cadaed1/scripts/tf_cnn_benchmarks/variable_mgr_util.py#L140-L166
class OverrideCachingDevice(object):
    """Variable getter which caches variables on the least loaded device.

    Variables smaller than a certain threshold are cached on a single specific
    device, as specified in the constructor. All other variables are load balanced
    across a pool of devices, by caching each variable on the least loaded device.
    """

    def __init__(self, devices, device_for_small_variables,
                 small_variable_size_threshold):
        self.devices = devices
        self.sizes = [0] * len(self.devices)
        self.device_for_small_variables = device_for_small_variables
        self.small_variable_size_threshold = small_variable_size_threshold

    def __call__(self, getter, *args, **kwargs):
        size = tf.TensorShape(kwargs['shape']).num_elements()
        if size is None or not kwargs.get('trainable', True):
            # TODO a lot of vars won't be saved then
            _replace_global_by_local(kwargs)
            return getter(*args, **kwargs)

        if size < self.small_variable_size_threshold:
            device_name = self.device_for_small_variables
        else:
            device_index, _ = min(enumerate(self.sizes), key=operator.itemgetter(1))
            device_name = self.devices[device_index]
            self.sizes[device_index] += size

        kwargs['caching_device'] = device_name
        var = getter(*args, **kwargs)
        return var
