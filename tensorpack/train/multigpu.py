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
from ..tfutils.tower import TowerContext
from ..tfutils.collection import backup_collection, restore_collection
from ..tfutils.gradproc import FilterNoneGrad, ScaleGradient

from .base import Trainer
from .feedfree import SingleCostFeedfreeTrainer
from .input_source import QueueInput, StagingInputWrapper

__all__ = ['SyncMultiGPUTrainer', 'AsyncMultiGPUTrainer']


class MultiGPUTrainer(Trainer):
    """ Base class for multi-gpu training"""
    @staticmethod
    def build_on_multi_tower(towers, func, devices=None):
        """
        Args:
            towers: list of gpu relative ids
            func: a lambda to be called inside each tower
            devices: a list of devices to be used. By default will use GPUs in towers.
        """
        logger.info("Training a model of {} tower".format(len(towers)))

        ret = []
        global_scope = tf.get_variable_scope()
        if devices is not None:
            assert len(devices) == len(towers)
        for idx, t in enumerate(towers):
            device = devices[idx] if devices is not None else '/gpu:{}'.format(t)
            with tf.variable_scope(global_scope, reuse=idx > 0), \
                TowerContext(
                    'tower{}'.format(idx),
                    device=device,
                    is_training=True):
                logger.info("Building graph for training tower {}...".format(idx))

                ret.append(func())

                if idx == 0:
                    # avoid repeated summary & update_ops from each device
                    backup = backup_collection(TOWER_FREEZE_KEYS)
        restore_collection(backup)
        return ret


# Copied from https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/variable_mgr.py
class ParamServerDeviceSetter(object):
    """Helper class to assign variables on the least loaded ps-device."""
    def __init__(self, worker_device, ps_devices):
        """
        Args:
            worker_device: the device to use for computer ops.
            ps_devices: a list of device to use for Variable ops. Each variable is
                assigned to the least loaded device.
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


class SyncMultiGPUTrainerParameterServer(MultiGPUTrainer, SingleCostFeedfreeTrainer):
    """
    A multi-tower multi-GPU trainer which synchronoizes the gradients computed
    from each tower, averages them and update to variables stored on PS.
    """

    def __init__(self, config, ps_device='gpu'):
        """
        Args:
            config: same as in :class:`QueueInputTrainer`.
            ps_device: either 'gpu' or 'cpu'
        """
        if config.dataflow is not None:
            # use queueinput by default. May need to avoid this in the future (when more input type is available)
            self._input_source = QueueInput(config.dataflow)
        else:
            self._input_source = config.data

        if len(config.tower) > 1:
            assert tf.test.is_gpu_available()

            # seem to only improve on >1 GPUs
            if not isinstance(self._input_source, StagingInputWrapper):
                devices = ['/gpu:{}'.format(k) for k in config.tower]
                self._input_source = StagingInputWrapper(self._input_source, devices)

        assert ps_device in ['gpu', 'cpu'], ps_device
        self._ps_device = ps_device
        super(SyncMultiGPUTrainerParameterServer, self).__init__(config)

    @staticmethod
    def _average_grads(tower_grads):
        nr_tower = len(tower_grads)
        if nr_tower == 1:
            return tower_grads[0]
        new_tower_grads = []
        with tf.name_scope('AvgGrad'):
            for grad_and_vars in zip(*tower_grads):
                # Ngpu * 2
                v = grad_and_vars[0][1]
                all_grads = [g for (g, _) in grad_and_vars]

                nones = list(set(all_grads))
                if None in nones and len(nones) != 1:
                    raise RuntimeError("Gradient w.r.t {} is None in some but not all towers!".format(v.name))
                elif nones[0] is None:
                    logger.warn("No Gradient w.r.t {}".format(v.op.name))
                    continue
                try:
                    with tf.device(v.device):       # colocate summed grad with var
                        grad = tf.multiply(tf.add_n(all_grads), 1.0 / nr_tower)
                except:
                    logger.error("Error while processing gradients of {}".format(v.name))
                    raise
                new_tower_grads.append((grad, v))
        return new_tower_grads

    def _setup(self):
        super(SyncMultiGPUTrainerParameterServer, self)._setup()

        raw_devices = ['/gpu:{}'.format(k) for k in self.config.tower]
        if self._ps_device == 'gpu':
            devices = [ParamServerDeviceSetter(d, raw_devices) for d in raw_devices]
        else:
            devices = [tf.train.replica_device_setter(
                worker_device=d, ps_device='/cpu:0', ps_tasks=1) for d in raw_devices]

        grad_list = MultiGPUTrainer.build_on_multi_tower(
            self.config.tower, lambda: self._get_cost_and_grad()[1], devices)

        # debug tower performance (without update):
        # ops = [k[0] for k in grad_list[1]] + [k[0] for k in grad_list[0]]
        # self.train_op = tf.group(*ops)
        # return

        grads = SyncMultiGPUTrainerParameterServer._average_grads(grad_list)
        # grads = grad_list[0]

        self.train_op = self.config.optimizer.apply_gradients(grads, name='min_op')


def SyncMultiGPUTrainer(config):
    """
    Alias for ``SyncMultiGPUTrainerParameterServer(config, ps_device='gpu')``,
    as this is the most commonly used synchronous multigpu trainer.
    """
    return SyncMultiGPUTrainerParameterServer(config, ps_device='gpu')


class AsyncMultiGPUTrainer(MultiGPUTrainer,
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
        if config.dataflow is not None:
            self._input_source = QueueInput(config.dataflow)
        else:
            self._input_source = config.data
        super(AsyncMultiGPUTrainer, self).__init__(config)

        self._scale_gradient = scale_gradient

        if len(config.tower) > 1:
            assert tf.test.is_gpu_available()

    def _setup(self):
        super(AsyncMultiGPUTrainer, self)._setup()
        grad_list = MultiGPUTrainer.build_on_multi_tower(
            self.config.tower, lambda: self._get_cost_and_grad()[1])
        grad_list = [FilterNoneGrad().process(gv) for gv in grad_list]
        if self._scale_gradient and self.config.nr_tower > 1:
            # pretend to average the grads, in order to make async and
            # sync have consistent effective learning rate
            gradproc = ScaleGradient(('.*', 1.0 / self.config.nr_tower), log=False)
            grad_list = [gradproc.process(gv) for gv in grad_list]

        # use grad from the first tower for iteration in main thread
        self.train_op = self.config.optimizer.apply_gradients(grad_list[0], name='min_op')

        self._start_async_threads(grad_list)

    def _start_async_threads(self, grad_list):
        # prepare train_op for the rest of the towers
        # itertools.count is atomic w.r.t. python threads
        self.async_step_counter = itertools.count()
        self.training_threads = []
        for k in range(1, self.config.nr_tower):
            train_op = self.config.optimizer.apply_gradients(grad_list[k])

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
