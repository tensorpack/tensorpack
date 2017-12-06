#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..callbacks.graph import RunOp
from ..utils.develop import log_deprecated

from ..input_source import QueueInput, StagingInput, DummyConstantInput
from ..graph_builder.training import (
    SyncMultiGPUParameterServerBuilder,
    SyncMultiGPUReplicatedBuilder,
    AsyncMultiGPUBuilder,
    DataParallelBuilder)
from .base import Trainer

__all__ = ['MultiGPUTrainerBase',
           'SyncMultiGPUTrainerReplicated',
           'SyncMultiGPUTrainerParameterServer',
           'AsyncMultiGPUTrainer',
           'SyncMultiGPUTrainer']


class MultiGPUTrainerBase(Trainer):
    """
    For backward compatibility only
    """
    def build_on_multi_tower(towers, func, devices=None, use_vs=None):
        log_deprecated("MultiGPUTrainerBase.build_on_multitower",
                       "Please use DataParallelBuilder.build_on_towers", "2018-01-31")
        return DataParallelBuilder.build_on_towers(towers, func, devices, use_vs)


def apply_prefetch_policy(config, gpu_prefetch=True):
    assert (config.data is not None or config.dataflow is not None) and config.model is not None
    if config.data is None and config.dataflow is not None:
        # always use Queue prefetch
        config.data = QueueInput(config.dataflow)
        config.dataflow = None
    if len(config.tower) > 1 and gpu_prefetch:
        assert tf.test.is_gpu_available()

        # seem to only improve on >1 GPUs
        if not isinstance(config.data, (StagingInput, DummyConstantInput)):
            config.data = StagingInput(config.data, config.tower)


class SyncMultiGPUTrainerParameterServer(Trainer):

    __doc__ = SyncMultiGPUParameterServerBuilder.__doc__

    def __init__(self, config, ps_device='gpu', gpu_prefetch=True):
        """
        Args:
            config(TrainConfig): Must contain 'model' and either one of 'data' or 'dataflow'.
            ps_device: either 'gpu' or 'cpu', where variables are stored.  Setting to 'cpu' might help when #gpu>=4
            gpu_prefetch(bool): whether to prefetch the data to each GPU. Usually improve performance.
        """
        apply_prefetch_policy(config, gpu_prefetch)
        self._input_source = config.data

        assert ps_device in ['gpu', 'cpu'], ps_device
        self._ps_device = ps_device
        super(SyncMultiGPUTrainerParameterServer, self).__init__(config)

    def _setup(self):
        callbacks = self._input_source.setup(self.model.get_inputs_desc())

        self.train_op = SyncMultiGPUParameterServerBuilder(
            self._config.tower, self._ps_device).build(
                lambda: self.model._build_graph_get_grads(
                    *self._input_source.get_input_tensors()),
                self.model.get_optimizer)

        self._config.callbacks.extend(callbacks)


def SyncMultiGPUTrainer(config):
    """
    Alias for ``SyncMultiGPUTrainerParameterServer(config, ps_device='gpu')``,
    as this is the most commonly used synchronous multigpu trainer (but may
    not be more efficient than the other).
    """
    return SyncMultiGPUTrainerParameterServer(config, ps_device='gpu')


class SyncMultiGPUTrainerReplicated(Trainer):

    __doc__ = SyncMultiGPUReplicatedBuilder.__doc__

    def __init__(self, config, gpu_prefetch=True):
        """
        Args:
            config, gpu_prefetch: same as in :class:`SyncMultiGPUTrainerParameterServer`
        """
        apply_prefetch_policy(config, gpu_prefetch)
        self._input_source = config.data
        super(SyncMultiGPUTrainerReplicated, self).__init__(config)

    def _setup(self):
        callbacks = self._input_source.setup(self.model.get_inputs_desc())

        self.train_op, post_init_op = SyncMultiGPUReplicatedBuilder(
            self._config.tower).build(
                lambda: self.model._build_graph_get_grads(
                    *self._input_source.get_input_tensors()),
                self.model.get_optimizer)

        cb = RunOp(
            lambda: post_init_op,
            run_before=True, run_as_trigger=True, verbose=True)
        self._config.callbacks.extend(callbacks + [cb])


class AsyncMultiGPUTrainer(Trainer):

    __doc__ = AsyncMultiGPUBuilder.__doc__

    def __init__(self, config, scale_gradient=True):
        """
        Args:
            config(TrainConfig): Must contain 'model' and either one of 'data' or 'dataflow'.
            scale_gradient (bool): if True, will scale each gradient by ``1.0/nr_gpu``.
        """
        apply_prefetch_policy(config)
        self._input_source = config.data
        self._scale_gradient = scale_gradient
        super(AsyncMultiGPUTrainer, self).__init__(config)

    def _setup(self):
        callbacks = self._input_source.setup(self.model.get_inputs_desc())

        self.train_op = AsyncMultiGPUBuilder(
            self._config.tower, self._scale_gradient).build(
                lambda: self.model._build_graph_get_grads(
                    *self._input_source.get_input_tensors()),
                self.model.get_optimizer)

        self._config.callbacks.extend(callbacks)
