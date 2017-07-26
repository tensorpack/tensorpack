#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import operator
from six.moves import zip, range

from ..utils import logger
from ..utils.naming import TOWER_FREEZE_KEYS
from ..tfutils.common import get_tf_version_number
from ..tfutils.tower import TowerContext
from ..tfutils.collection import backup_collection, restore_collection
from ..tfutils.gradproc import ScaleGradient
from ..callbacks.graph import RunOp

from ..graph_builder.input_source import QueueInput, StagingInputWrapper, DummyConstantInput
from .feedfree import FeedfreeTrainerBase

__all__ = ['MultiGPUTrainerBase', 'SyncMultiGPUTrainer',
           'AsyncMultiGPUTrainer', 'LeastLoadedDeviceSetter',
           'SyncMultiGPUTrainerReplicated',
           'SyncMultiGPUTrainerParameterServer']


def _check_tf_version():
    assert get_tf_version_number() >= 1.1, \
        "TF version {} is too old to run multi GPU training!".format(tf.VERSION)


def apply_prefetch_policy(config, gpu_prefetch=True):
    if config.data is None and config.dataflow is not None:
        # always use Queue prefetch
        config.data = QueueInput(config.dataflow)
        config.dataflow = None
    if len(config.tower) > 1 and gpu_prefetch:
        assert tf.test.is_gpu_available()

        # seem to only improve on >1 GPUs
        if not isinstance(config.data, (StagingInputWrapper, DummyConstantInput)):
            devices = ['/gpu:{}'.format(k) for k in config.tower]
            config.data = StagingInputWrapper(config.data, devices)


class MultiGPUTrainerBase(FeedfreeTrainerBase):
    """ Base class for multi-gpu training"""
    @staticmethod
    def build_on_multi_tower(
            towers, func,
            devices=None, var_strategy='shared',
            vs_names=None):
        """
        Args:
            towers: list of gpu relative ids
            func: a lambda to be called inside each tower
            devices: a list of devices to be used. By default will use GPUs in ``towers``.
            var_strategy (str): 'shared' or 'replicated'
            vs_names (list[str]): list of variable scope names to use.

        Returns:
            List of outputs of ``func``, evaluated on each tower.
        """
        logger.info("Training a model of {} tower".format(len(towers)))
        if len(towers) > 1:
            _check_tf_version()

        ret = []
        if devices is not None:
            assert len(devices) == len(towers)

        tower_names = ['tower{}'.format(idx) for idx in range(len(towers))]
        keys_to_freeze = TOWER_FREEZE_KEYS[:]
        if var_strategy == 'replicated':        # TODO ugly
            logger.info("In replicated mode, UPDATE_OPS from all GPUs will be run.")
            keys_to_freeze.remove(tf.GraphKeys.UPDATE_OPS)
            # fix all Nones. TODO ugly
            if vs_names is not None:
                assert len(vs_names) == len(towers)
                for idx, name in enumerate(vs_names):
                    if name is None:
                        vs_names[idx] = tower_names[idx]
            else:
                vs_names = tower_names
        else:
            assert vs_names is None
            vs_names = [''] * len(towers)

        for idx, t in enumerate(towers):
            device = devices[idx] if devices is not None else '/gpu:{}'.format(t)
            with tf.device(device), TowerContext(
                    tower_names[idx],
                    is_training=True,
                    index=idx,
                    vs_name=vs_names[idx]):
                if idx == t:
                    logger.info("Building graph for training tower {}...".format(idx))
                else:
                    logger.info("Building graph for training tower {} on device {}...".format(idx, device))

                ret.append(func())

                if idx == 0:
                    # avoid duplicated summary & update_ops from each device
                    backup = backup_collection(keys_to_freeze)
        restore_collection(backup)
        return ret

    @staticmethod
    def _check_grad_list(grad_list):
        """
        Args:
            grad_list: list of list of tuples, shape is Ngpu x Nvar x 2
        """
        nvars = [len(k) for k in grad_list]
        assert len(set(nvars)) == 1, "Number of gradients from each tower is different! " + str(nvars)

    @staticmethod
    def _build_graph_get_grads(model, input):
        model.build_graph(input)
        return model.get_cost_and_grad()[1]


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


class SyncMultiGPUTrainerParameterServer(MultiGPUTrainerBase):
    """
    A data-parallel Multi-GPU trainer which synchronoizes the gradients computed
    from each tower, averages them and update to variables stored across all
    GPUs or on CPU.
    """

    def __init__(self, config, ps_device='gpu', gpu_prefetch=True):
        """
        Args:
            config(TrainConfig):
            ps_device: either 'gpu' or 'cpu', where variables are stored.
            gpu_prefetch(bool): whether to prefetch the data to each GPU. Usually improve performance.
        """
        apply_prefetch_policy(config, gpu_prefetch)
        self._input_source = config.data

        assert ps_device in ['gpu', 'cpu'], ps_device
        self._ps_device = ps_device
        super(SyncMultiGPUTrainerParameterServer, self).__init__(config)

    @staticmethod
    def _average_grads(tower_grads):
        # tower_grads: Ngpu x Nvar x 2
        nr_tower = len(tower_grads)
        if nr_tower == 1:
            return tower_grads[0]
        new_tower_grads = []
        with tf.name_scope('AvgGrad'):
            for grad_and_vars in zip(*tower_grads):
                # Ngpu * 2
                v = grad_and_vars[0][1]
                all_grads = [g for (g, _) in grad_and_vars]

                with tf.device(v.device):       # colocate summed grad with var
                    grad = tf.multiply(
                        tf.add_n(all_grads), 1.0 / nr_tower)
                    new_tower_grads.append((grad, v))
        return new_tower_grads

    def _setup(self):
        super(SyncMultiGPUTrainerParameterServer, self)._setup()

        raw_devices = ['/gpu:{}'.format(k) for k in self.config.tower]
        if self._ps_device == 'gpu':
            devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        else:
            devices = [tf.train.replica_device_setter(
                worker_device=d, ps_device='/cpu:0', ps_tasks=1) for d in raw_devices]

        grad_list = MultiGPUTrainerBase.build_on_multi_tower(
            self.config.tower,
            lambda: MultiGPUTrainerBase._build_graph_get_grads(
                self.model, self._input_source), devices)
        MultiGPUTrainerBase._check_grad_list(grad_list)

        # debug tower performance (without update):
        # ops = [k[0] for k in grad_list[1]] + [k[0] for k in grad_list[0]]
        # self.train_op = tf.group(*ops)
        # return

        grads = self._average_grads(grad_list)
        # grads = grad_list[0]

        self.train_op = self.model.get_optimizer().apply_gradients(
            grads, name='train_op')


def SyncMultiGPUTrainer(config):
    """
    Alias for ``SyncMultiGPUTrainerParameterServer(config, ps_device='gpu')``,
    as this is the most commonly used synchronous multigpu trainer (but may
    not be more efficient than the other).
    """
    return SyncMultiGPUTrainerParameterServer(config, ps_device='gpu')


class SyncMultiGPUTrainerReplicated(MultiGPUTrainerBase):
    """
    Data-parallel Multi-GPU trainer where each GPU contains a replicate of the
    whole model. Each gradient update is broadcast and synced.
    """
    def __init__(self, config, gpu_prefetch=True):
        """
        Args:
            config, gpu_prefetch: same as in :class:`SyncMultiGPUTrainerParameterServer`
        """
        apply_prefetch_policy(config, gpu_prefetch)
        self._input_source = config.data
        logger.warn("Note that SyncMultiGPUTrainerReplicated doesn't support inference.")
        super(SyncMultiGPUTrainerReplicated, self).__init__(config)

    @staticmethod
    def _allreduce_grads(tower_grads):
        from tensorflow.contrib import nccl
        nr_tower = len(tower_grads)
        if nr_tower == 1:
            return [[x] for x in tower_grads[0]]
        new_tower_grads = []
        with tf.name_scope('AvgGrad'):
            for grad_and_vars in zip(*tower_grads):
                v = grad_and_vars[0][1]
                grads = [g for g, _ in grad_and_vars]
                summed = nccl.all_sum(grads)

                grads_for_a_var = []
                for (_, v), g in zip(grad_and_vars, summed):
                    with tf.device(g.device):
                        g = tf.multiply(g, 1.0 / nr_tower)
                        grads_for_a_var.append((g, v))
                new_tower_grads.append(grads_for_a_var)
        # NVar * NGPU * 2
        return new_tower_grads

    def _setup(self):
        super(SyncMultiGPUTrainerReplicated, self)._setup()
        raw_devices = ['/gpu:{}'.format(k) for k in self.config.tower]

        grad_list = MultiGPUTrainerBase.build_on_multi_tower(
            self.config.tower,
            lambda: MultiGPUTrainerBase._build_graph_get_grads(
                self.model, self._input_source),
            var_strategy='replicated',
            # use no variable scope for the first tower
            vs_names=[''] + [None] * (self.config.nr_tower - 1))
        grads = self._allreduce_grads(grad_list)

        train_ops = []
        opt = self.model.get_optimizer()
        for idx in range(self.config.nr_tower):
            with tf.device(raw_devices[idx]):
                grad_and_vars = [x[idx] for x in grads]
                train_ops.append(opt.apply_gradients(
                    grad_and_vars, name='apply_grad_{}'.format(idx)))
        self.train_op = tf.group(*train_ops, name='train_op')
        self.register_callback(RunOp(
            SyncMultiGPUTrainerReplicated.get_post_init_ops,
            run_before=True, run_as_trigger=True, verbose=True))


# Adopt from https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/variable_mgr.py
    @staticmethod
    def get_post_init_ops():
        # Copy initialized values for variables on GPU 0 to other GPUs.
        all_vars = tf.trainable_variables()
        all_vars.extend(tf.model_variables())
        var_by_name = dict([(v.name, v) for v in all_vars])
        post_init_ops = []
        for v in all_vars:
            split_name = v.name.split('/')
            if not v.name.startswith('tower'):
                continue
            if v.name.startswith('tower0'):
                continue        # TODO some vars (EMA) may still startswith tower0
            # in this trainer, the master name doesn't have the towerx/ prefix
            split_name = split_name[1:]
            copy_from = var_by_name['/'.join(split_name)]
            post_init_ops.append(v.assign(copy_from.read_value()))
        logger.info(
            "'sync_variables_from_tower0' includes {} operations.".format(len(post_init_ops)))
        return tf.group(*post_init_ops, name='sync_variables_from_tower0')


class AsyncMultiGPUTrainer(MultiGPUTrainerBase):
    """
    A multi-tower multi-GPU trainer where each tower independently
    asynchronously updates the model without averaging the gradient.
    """

    def __init__(self, config, scale_gradient=True):
        """
        Args:
            config(TrainConfig):
            scale_gradient (bool): if True, will scale each gradient by ``1.0/nr_gpu``.
        """
        apply_prefetch_policy(config)
        self._input_source = config.data
        self._scale_gradient = scale_gradient
        super(AsyncMultiGPUTrainer, self).__init__(config)

    def _setup(self):
        super(AsyncMultiGPUTrainer, self)._setup()
        raw_devices = ['/gpu:{}'.format(k) for k in self.config.tower]
        devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        grad_list = MultiGPUTrainerBase.build_on_multi_tower(
            self.config.tower,
            lambda: MultiGPUTrainerBase._build_graph_get_grads(
                self.model, self._input_source), devices)
        MultiGPUTrainerBase._check_grad_list(grad_list)

        if self._scale_gradient and self.config.nr_tower > 1:
            # pretend to average the grads, in order to make async and
            # sync have consistent effective learning rate
            gradproc = ScaleGradient(('.*', 1.0 / self.config.nr_tower), verbose=False)
            grad_list = [gradproc.process(gv) for gv in grad_list]
        # Ngpu x Nvar x 2

        train_ops = []
        opt = self.model.get_optimizer()
        for i, grad_and_vars in enumerate(zip(*grad_list)):
            # Ngpu x 2
            v = grad_and_vars[0][1]
            with tf.device(v.device):
                # will call apply_gradients (therefore gradproc) multiple times
                train_ops.append(opt.apply_gradients(
                    grad_and_vars, name='apply_grad_{}'.format(i)))
        self.train_op = tf.group(*train_ops, name='train_op')
