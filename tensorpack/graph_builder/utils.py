# -*- coding: utf-8 -*-
# File: utils.py


import operator
from contextlib import contextmanager
import tensorflow as tf
import threading

from ..compat import tfv1
from ..tfutils.common import get_tf_version_tuple
from ..tfutils.scope_utils import cached_name_scope, under_name_scope
from ..tfutils.varreplace import custom_getter_scope
from ..utils import logger
from ..utils.argtools import call_only_once

__all__ = ["LeastLoadedDeviceSetter", "allreduce_grads"]


"""
Some utilities for building the graph.
"""


def _replace_global_by_local(kwargs):
    if 'collections' in kwargs:
        collections = kwargs['collections']
    if not collections:
        collections = {tfv1.GraphKeys.GLOBAL_VARIABLES}
    else:
        collections = set(collections.copy())
    collections.remove(tfv1.GraphKeys.GLOBAL_VARIABLES)
    collections.add(tfv1.GraphKeys.LOCAL_VARIABLES)
    kwargs['collections'] = list(collections)


_module_lock = threading.Lock()
_shared_cnt_counter = 0


def _get_shared_cnt():
    global _shared_cnt_counter

    with _module_lock:
        val = _shared_cnt_counter
        _shared_cnt_counter += 1
    return val


@contextmanager
def override_to_local_variable(enable=True):
    """
    Returns:
        a context where all variables will be created as local.
    """
    if enable:

        def custom_getter(getter, name, *args, **kwargs):
            _replace_global_by_local(kwargs)
            return getter(name, *args, **kwargs)

        with custom_getter_scope(custom_getter):
            yield
    else:
        yield


# https://github.com/tensorflow/benchmarks/blob/48cbef14a592e02a14beee8e9aef3ad22cadaed1/scripts/tf_cnn_benchmarks/variable_mgr_util.py#L192-L218
class LeastLoadedDeviceSetter(object):
    """
    Helper class to assign variables on the least loaded ps-device.

    Usage:

        .. code-block:: python

            with tf.device(LeastLoadedDeviceSetter(...)):
                ...
    """
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
        # from tensorflow.python.training.device_util import canonicalize
        # from tensorflow.python.distribute.device_util import canonicalize
        def canonicalize(name):    # tensorflow/tensorflow#11484
            return tfv1.DeviceSpec.from_string(name).to_string()

        if op.device:
            return op.device
        if op.type not in ['Variable', 'VariableV2']:
            return canonicalize(self.worker_device)

        device_name = self.place_with_balance(op)
        return canonicalize(device_name)

    def place_with_balance(self, op):
        device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]
        var_size = op.outputs[0].get_shape().num_elements()
        if var_size is None:
            logger.warn("[LeastLoadedDeviceSetter] Shape of variable {} is not fully defined!".format(op.name))
            var_size = 0
        self.ps_sizes[device_index] += var_size
        return device_name

    def __str__(self):
        return "LeastLoadedDeviceSetter-{}".format(self.worker_device)


def split_grad_list(grad_list):
    """
    Args:
        grad_list: K x N x 2

    Returns:
        K x N: gradients
        K x N: variables
    """
    g = []
    v = []
    for tower in grad_list:
        g.append([x[0] for x in tower])
        v.append([x[1] for x in tower])
    return g, v


def merge_grad_list(all_grads, all_vars):
    """
    Args:
        all_grads (K x N): gradients
        all_vars(K x N): variables

    Return:
        K x N x 2: list of list of (grad, var) pairs
    """
    return [list(zip(gs, vs)) for gs, vs in zip(all_grads, all_vars)]


@under_name_scope('AllReduceGrads')
def allreduce_grads(all_grads, average, mode="nccl"):
    """
    All-reduce average the gradients among K devices. Results are broadcasted to all devices.

    Args:
        all_grads (K x N): List of list of gradients. N is the number of variables.
        average (bool): average gradients or not.
        mode (str): "nccl", "collective"

    Returns:
        K x N: same as input, but each grad is replaced by the average over K devices.
    """
    assert mode in ["nccl", "collective"], mode

    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads
    new_all_grads = []  # N x K
    for grads in zip(*all_grads):
        # k grads
        if mode == "nccl":
            if get_tf_version_tuple() <= (1, 12):
                from tensorflow.contrib import nccl  # deprecated
            else:
                from tensorflow.python.ops import nccl_ops as nccl
            summed = nccl.all_sum(grads)
        else:
            from tensorflow.python.ops import collective_ops
            summed = []
            shared_cnt = _get_shared_cnt()
            for t in grads:
                with tf.device(t.device):
                    t = collective_ops.all_reduce(
                        t, len(grads),
                        42,   # group key is any fixed integer for a fixed group of devices
                        shared_cnt + 100,
                        'Add', 'Id', communication_hint='nccl')
                    summed.append(t)

        grads_for_devices = []  # K
        for g in summed:
            with tf.device(g.device):
                # tensorflow/benchmarks didn't average gradients
                if average:
                    g = tf.multiply(g, 1.0 / nr_tower)
            grads_for_devices.append(g)
        new_all_grads.append(grads_for_devices)

    # transpose to K x N
    ret = list(zip(*new_all_grads))
    return ret


@under_name_scope('AllReduceGradsHierachical')
def allreduce_grads_hierarchical(all_grads, devices, average=False):
    """
    Hierarchical allreduce for DGX-1 system.

    Args:
        all_grads (K x N): List of list of gradients. N is the number of variables.
        devices ([str]): K str for the K devices.
        average (bool): average gradients or not.

    Returns:
        (K x N): same as input, but each grad is replaced by the average over K lists.
    """
    num_gpu = len(devices)
    assert num_gpu == 8, num_gpu
    assert len(all_grads) == num_gpu, len(all_grads)
    group_size = num_gpu // 2

    agg_all_grads = []  # N x K
    for varid, grads in enumerate(zip(*all_grads)):
        # grads: K gradients
        g0_main_gpu = varid % num_gpu
        g1_main_gpu = (g0_main_gpu + group_size) % num_gpu
        g0_start = 0 if g0_main_gpu < group_size else group_size
        g1_start = 0 if g1_main_gpu < group_size else group_size
        assert g0_start != g1_start
        g0_grads = grads[g0_start: g0_start + group_size]
        g1_grads = grads[g1_start: g1_start + group_size]

        with tf.device(devices[g0_main_gpu]):
            g0_agg = tf.add_n(g0_grads, name='group0_agg')

        with tf.device(devices[g1_main_gpu]):
            g1_agg = tf.add_n(g1_grads, name='group1_agg')
            g1_total_agg = tf.add(g0_agg, g1_agg, name='group1_total_agg')

        with tf.device(devices[g0_main_gpu]):
            g0_total_agg = tf.identity(g1_total_agg, name='group0_total_agg')

        agg_grads = []  # K aggregated grads
        for k in range(num_gpu):
            if (k < group_size) == (g0_main_gpu < group_size):
                main_gpu = g0_total_agg
            else:
                main_gpu = g1_total_agg
            with tf.device(devices[k]):
                if not average:
                    device_total_agg = tf.identity(
                        main_gpu, name='device{}_total_agg'.format(k))
                else:
                    # TODO where to put average?
                    device_total_agg = tf.multiply(
                        main_gpu, 1.0 / num_gpu, name='device{}_total_agg'.format(k))
                agg_grads.append(device_total_agg)

        agg_all_grads.append(agg_grads)

    # transpose
    agg_all_grads = list(zip(*agg_all_grads))   # K x Nvar
    return agg_all_grads


@under_name_scope('AggregateGradsColocate')
def aggregate_grads_colocate(all_grads, average=True):
    """
    Aggregate the gradients. The aggregation is colocated with the variable.

    Args:
        all_grads (K x N x 2): A list of K lists. Each of the list is a list of N (grad, var) tuples.
            The variables have to be shared across the K lists.
        average (bool): do average or sum
    Returns:
        (N x 2): A list of N (grad, var) tuples, where grad is averaged or summed over K.
    """
    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads[0]

    def aggregate(grads):
        if average:
            return tf.multiply(tf.add_n(grads), 1.0 / nr_tower)
        else:
            return tf.add_n(grads)

    ret = []
    for idx, grad_and_vars in enumerate(zip(*all_grads)):
        # Ngpu * 2
        v = grad_and_vars[0][1]
        grads = [g for (g, _) in grad_and_vars]
        with tf.device(v.device):       # colocate summed grad with var
            grad = aggregate(grads)
        ret.append((grad, v))
    return ret


@under_name_scope('AllReduceNaive')
def allreduce_grads_naive(all_grads, devices=None, average=True):
    """
    AllReduce the gradients with raw ops (instead of collective ops).

    Args:
        all_grads (K x N): A list of K lists. Each of the list is a list of N grad tuples.
            The variables have to be the same across the K lists.
        devices (list[str]): assign the averaging to these device in
            round-robin. Cannot be used together with ``colocation``.
        average (bool): do average or sum

    Returns:
        list[Tensor]: list of grads where each grad is averaged or summed over K.
    """
    if devices is not None:
        assert isinstance(devices, list), devices
        # device_setter = LeastLoadedDeviceSetter(None, devices)

    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads[0]

    def aggregate(grads):
        if average:
            return tf.multiply(tf.add_n(grads), 1.0 / nr_tower)
        else:
            return tf.add_n(grads)

    grads_ret = []  # N(rev) grads
    # reverse so the device placement makes the last part of model more balance?
    all_grads_rev = [x[::-1] for x in all_grads]   # K x N(rev)

    for idx, grads in enumerate(zip(*all_grads_rev)):
        # grads: K tensors
        if devices is None:
            grad = aggregate(grads)
        else:
            # dev = device_setter.place_with_balance(v.op)
            dev = devices[idx % len(devices)]
            with tf.device(dev):
                grad = aggregate(grads)
        grads_ret.append(grad)
    grads_ret = grads_ret[::-1]
    return grads_ret


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


# TODO pack at variable boundary, so that the concat does not have to wait for all
# grads to be ready
class GradientPacker(object):
    """
    Concat gradients together to optimize transfer.
    """

    def __init__(self, num_split=8):
        self._num_split = num_split

    @call_only_once
    def compute_strategy(self, grads):
        """
        Returns:
            bool - False if grads cannot be packed due to various reasons.
        """
        for g in grads:
            if not g.shape.is_fully_defined():
                logger.warn("Found gradient with incomplete shape: "
                            "{} has shape {}".format(g.name, g.shape))
                return False

        self._shapes = [g.shape for g in grads]
        self._sizes = [g.shape.num_elements() for g in grads]
        self._total_size = sum(self._sizes)
        if self._total_size / self._num_split < 1024:
            logger.info("Skip GradientPacker due to too few gradients.")
            return False
        # should have the same dtype
        dtypes = {g.dtype for g in grads}
        if len(dtypes) != 1:
            logger.info("Skip GradientPacker due to inconsistent gradient types.")
            return False
        self._grad_dtype = grads[0].dtype

        split_size = self._total_size // self._num_split
        split_size_last = self._total_size - split_size * (self._num_split - 1)
        self._split_sizes = [split_size] * (self._num_split - 1) + [split_size_last]
        logger.info(
            "Will pack {} gradients of total dimension={} into {} splits.".format(
                len(self._sizes), self._total_size, self._num_split))
        return True

    def pack(self, grads):
        """
        Args:
            grads (list): list of gradient tensors

        Returns:
            packed list of gradient tensors to be aggregated.
        """
        for i, g in enumerate(grads):
            assert g.shape == self._shapes[i]

        with cached_name_scope("GradientPacker", top_level=False):
            concat_grads = tf.concat([tf.reshape(g, [-1]) for g in grads], 0, name='concatenated_grads')
            # concat_grads = tf.cast(concat_grads, tf.float16)
            grad_packs = tf.split(concat_grads, self._split_sizes)
            return grad_packs

    def unpack(self, grad_packs):
        with cached_name_scope("GradientPacker", top_level=False):
            concat_grads = tf.concat(grad_packs, 0, name='concatenated_packs')
            # concat_grads = tf.cast(concat_grads, self._grad_dtype)
            flattened_grads = tf.split(concat_grads, self._sizes)
            grads = [tf.reshape(g, shape) for g, shape in zip(flattened_grads, self._shapes)]
            return grads

    def pack_all(self, all_grads, devices):
        """
        Args:
            all_grads: K x N, K lists of gradients to be packed
        """
        ret = []    # #GPU x #split
        for dev, grads in zip(devices, all_grads):
            with tf.device(dev):
                ret.append(self.pack(grads))
        return ret

    def unpack_all(self, all_packed, devices):
        """
        Args:
            all_packed: K lists of packed gradients.
        """
        all_grads = []  # #GPU x #Var
        for dev, packed_grads_single_device in zip(devices, all_packed):
            with tf.device(dev):
                all_grads.append(self.unpack(packed_grads_single_device))
        return all_grads
