#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import itertools
import operator
import re
from six.moves import zip, range

from ..utils import logger
from ..utils.naming import TOWER_FREEZE_KEYS
from ..utils.concurrency import LoopThread
from ..tfutils.common import get_tf_version_number
from ..tfutils.tower import TowerContext
from ..tfutils.collection import backup_collection, restore_collection
from ..tfutils.gradproc import FilterNoneGrad, ScaleGradient
from ..callbacks.graph import RunOp

from .base import Trainer
from .feedfree import SingleCostFeedfreeTrainer
from .input_source import QueueInput, StagingInputWrapper

__all__ = ['MultiGPUTrainerBase', 'SyncMultiGPUTrainer',
           'AsyncMultiGPUTrainer', 'LeastLoadedDeviceSetter',
           'SyncMultiGPUTrainerReplicated',
           'SyncMultiGPUTrainerParameterServer']


def _check_tf_version():
    assert get_tf_version_number() >= 1.1, \
        "TF version {} is too old to run multi GPU training!".format(tf.VERSION)


def apply_prefetch_policy(config, use_stage=True):
    if config.data is None and config.dataflow is not None:
        config.data = QueueInput(config.dataflow)
        config.dataflow = None
    if len(config.tower) > 1 and use_stage:
        assert tf.test.is_gpu_available()

        # seem to only improve on >1 GPUs
        if not isinstance(config.data, StagingInputWrapper):
            devices = ['/gpu:{}'.format(k) for k in config.tower]
            config.data = StagingInputWrapper(config.data, devices)


class MultiGPUTrainerBase(Trainer):
    """ Base class for multi-gpu training"""
    @staticmethod
    def build_on_multi_tower(towers, func, devices=None, var_strategy='shared'):
        """
        Args:
            towers: list of gpu relative ids
            func: a lambda to be called inside each tower
            devices: a list of devices to be used. By default will use GPUs in towers.
            var_strategy (str):

        Returns:
            List of outputs of ``func``, evaluated on each tower.
        """
        logger.info("Training a model of {} tower".format(len(towers)))
        if len(towers) > 1:
            _check_tf_version()

        ret = []
        if devices is not None:
            assert len(devices) == len(towers)

        keys_to_freeze = TOWER_FREEZE_KEYS[:]
        if var_strategy == 'replicated':        # TODO ugly
            logger.info("UPDATE_OPS from all GPUs will be kept in the collection.")
            keys_to_freeze.remove(tf.GraphKeys.UPDATE_OPS)

        for idx, t in enumerate(towers):
            device = devices[idx] if devices is not None else '/gpu:{}'.format(t)
            with TowerContext(
                    'tower{}'.format(idx),
                    device=device, is_training=True,
                    var_strategy=var_strategy):
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
    def check_none_grads(name, grads):
        # grads: list of N grads
        nones = list(set(grads))
        if None in nones:
            if len(nones) != 1:
                raise RuntimeError("Gradient w.r.t {} is None in some but not all towers!".format(name))
            else:
                logger.warn("No Gradient w.r.t {}".format(name))
                return False
        return True


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
        if op.device:
            return op.device
        if op.type not in ['Variable', 'VariableV2']:
            return self.worker_device

        device_index, _ = min(enumerate(
            self.ps_sizes), key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]
        var_size = op.outputs[0].get_shape().num_elements()
        self.ps_sizes[device_index] += var_size

        return device_name


class SyncMultiGPUTrainerParameterServer(MultiGPUTrainerBase, SingleCostFeedfreeTrainer):
    """
    A data-parallel Multi-GPU trainer which synchronoizes the gradients computed
    from each tower, averages them and update to variables stored across all
    GPUs or on CPU.
    """

    def __init__(self, config, ps_device='gpu'):
        """
        Args:
            config: same as in :class:`QueueInputTrainer`.
            ps_device: either 'gpu' or 'cpu', where variables are stored.
        """
        apply_prefetch_policy(config)
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

                if not MultiGPUTrainerBase.check_none_grads(v.op.name, all_grads):
                    continue
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
            self.config.tower, lambda: self._get_cost_and_grad()[1], devices)

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
    as this is the most commonly used synchronous multigpu trainer.
    """
    return SyncMultiGPUTrainerParameterServer(config, ps_device='gpu')


class SyncMultiGPUTrainerReplicated(MultiGPUTrainerBase, SingleCostFeedfreeTrainer):
    """
    Data-parallel Multi-GPU trainer where each GPU contains a replicate of the
    whole model. Each gradient update is broadcast and synced.
    """
    def __init__(self, config):
        apply_prefetch_policy(config)
        self._input_source = config.data
        logger.warn("Note that SyncMultiGPUTrainerReplicated doesn't support inference.")
        super(SyncMultiGPUTrainerReplicated, self).__init__(config)

    @staticmethod
    def _allreduce_grads(tower_grads):
        from tensorflow.contrib import nccl
        nr_tower = len(tower_grads)
        if nr_tower == 1:
            return tower_grads[0]
        new_tower_grads = []
        with tf.name_scope('AvgGrad'):
            for grad_and_vars in zip(*tower_grads):
                v = grad_and_vars[0][1]
                grads = [g for g, _ in grad_and_vars]
                if not MultiGPUTrainerBase.check_none_grads(v.op.name, grads):
                    continue
                summed = nccl.all_sum(grads)

                grads_for_a_var = []
                for (_, v), g in zip(grad_and_vars, summed):
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
            lambda: self._get_cost_and_grad()[1],
            var_strategy='replicated')
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
            run_before=True, run_as_trigger=True))


# Adopt from https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/variable_mgr.py
    @staticmethod
    def get_post_init_ops():
        # Copy initialized values for variables on GPU 0 to other GPUs.
        global_vars = tf.global_variables()
        var_by_name = dict([(v.name, v) for v in global_vars])
        post_init_ops = []
        for v in global_vars:
            split_name = v.name.split('/')
            if not v.name.startswith('tower'):
                continue
            # the master name doesn't have the towerx/ prefix
            split_name = split_name[1:]
            copy_from = var_by_name['/'.join(split_name)]
            post_init_ops.append(v.assign(copy_from.read_value()))
        return tf.group(*post_init_ops, name='init_sync_vars')


class AsyncMultiGPUTrainer(MultiGPUTrainerBase,
                           SingleCostFeedfreeTrainer):
    """
    A multi-tower multi-GPU trainer where each tower independently
    asynchronously updates the model without locking.
    """

    def __init__(self, config,
                 scale_gradient=True):
        """
        Args:
            config: same as in :class:`QueueInputTrainer`.
            scale_gradient (bool): if True, will scale each gradient by
                ``1.0/nr_tower``, to make Async and Sync Trainer have the same
                effective learning rate.
        """
        apply_prefetch_policy(config, use_stage=False)
        logger.warn("Async training hasn't been well optimized. Sync training is even faster")
        self._input_source = config.data
        super(AsyncMultiGPUTrainer, self).__init__(config)

        self._scale_gradient = scale_gradient

        if len(config.tower) > 1:
            assert tf.test.is_gpu_available()

    def _setup(self):
        super(AsyncMultiGPUTrainer, self)._setup()
        grad_list = MultiGPUTrainerBase.build_on_multi_tower(
            self.config.tower, lambda: self._get_cost_and_grad()[1])
        grad_list = [FilterNoneGrad().process(gv) for gv in grad_list]
        if self._scale_gradient and self.config.nr_tower > 1:
            # pretend to average the grads, in order to make async and
            # sync have consistent effective learning rate
            gradproc = ScaleGradient(('.*', 1.0 / self.config.nr_tower), log=False)
            grad_list = [gradproc.process(gv) for gv in grad_list]

        # use grad from the first tower for iteration in main thread
        self._opt = self.model.get_optimizer()
        self.train_op = self._opt.apply_gradients(grad_list[0], name='min_op')

        self._start_async_threads(grad_list)

    def _start_async_threads(self, grad_list):
        # prepare train_op for the rest of the towers
        # itertools.count is atomic w.r.t. python threads
        self.async_step_counter = itertools.count()
        self.training_threads = []
        for k in range(1, self.config.nr_tower):
            train_op = self._opt.apply_gradients(grad_list[k])

            def f(op=train_op):  # avoid late-binding
                self.sess.run([op])         # TODO this won't work with StageInput
                next(self.async_step_counter)   # atomic due to GIL
            th = LoopThread(f)
            th.name = "AsyncLoopThread-{}".format(k)
            th.pause()
            th.start()
            self.training_threads.append(th)
        self.async_running = False

    def run_step(self):
        if not self.async_running:
            self.async_running = True
            for th in self.training_threads:  # resume all threads
                th.resume()
        next(self.async_step_counter)
        return super(AsyncMultiGPUTrainer, self).run_step()

    def _trigger_epoch(self):
        self.async_running = False
        for th in self.training_threads:
            th.pause()
        try:
            if self.config.nr_tower > 1:
                async_step_total_cnt = int(re.findall(
                    '[0-9]+', self.async_step_counter.__str__())[0])
                self.monitors.put(
                    'async_global_step', async_step_total_cnt)
        except:
            logger.exception("Cannot log async_global_step")
        super(AsyncMultiGPUTrainer, self)._trigger_epoch()
