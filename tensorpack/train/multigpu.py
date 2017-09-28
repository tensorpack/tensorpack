#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from six.moves import zip, range

from ..utils import logger
from ..utils.naming import TOWER_FREEZE_KEYS
from ..tfutils.common import get_tf_version_number
from ..tfutils.tower import TowerContext
from ..tfutils.collection import backup_collection, restore_collection
from ..tfutils.gradproc import ScaleGradient
from ..callbacks.graph import RunOp

from ..graph_builder.input_source import QueueInput, StagingInputWrapper, DummyConstantInput
from .base import Trainer
from .utility import LeastLoadedDeviceSetter, override_to_local_variable

__all__ = ['MultiGPUTrainerBase', 'LeastLoadedDeviceSetter',
           'SyncMultiGPUTrainerReplicated',
           'SyncMultiGPUTrainerParameterServer',
           'AsyncMultiGPUTrainer',
           'SyncMultiGPUTrainer']


def _check_tf_version():
    assert get_tf_version_number() >= 1.1, \
        "TF version {} is too old to run multi GPU training!".format(tf.VERSION)


def apply_prefetch_policy(config, gpu_prefetch=True):
    assert (config.data is not None or config.dataflow is not None) and config.model is not None
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


class MultiGPUTrainerBase(Trainer):
    """ Base class for multi-gpu training"""
    @staticmethod
    def build_on_multi_tower(
            towers, func,
            devices=None,
            use_vs=None):
        """
        Args:
            towers: list of gpu relative ids
            func: a lambda to be called inside each tower
            devices: a list of devices to be used. By default will use GPUs in ``towers``.
            use_vs (list[bool]): list of use_vs to passed to TowerContext

        Returns:
            List of outputs of ``func``, evaluated on each tower.
        """
        if len(towers) > 1:
            logger.info("Training a model of {} towers".format(len(towers)))
            _check_tf_version()

        ret = []
        if devices is not None:
            assert len(devices) == len(towers)
        if use_vs is not None:
            assert len(use_vs) == len(towers)

        tower_names = ['tower{}'.format(idx) for idx in range(len(towers))]
        keys_to_freeze = TOWER_FREEZE_KEYS[:]

        for idx, t in enumerate(towers):
            device = devices[idx] if devices is not None else '/gpu:{}'.format(t)
            usevs = use_vs[idx] if use_vs is not None else False
            with tf.device(device), TowerContext(
                    tower_names[idx],
                    is_training=True,
                    index=idx,
                    use_vs=usevs):
                if idx == t:
                    logger.info("Building graph for training tower {}...".format(idx))
                else:
                    logger.info("Building graph for training tower {} on device {}...".format(idx, device))

                # When use_vs is True, use LOCAL_VARIABLES,
                # so these duplicated variables won't be saved by default.
                with override_to_local_variable(enable=usevs):
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


class SyncMultiGPUTrainerParameterServer(MultiGPUTrainerBase):
    """
    A data-parallel multi-GPU trainer. It builds one tower on each GPU with
    shared variable scope. It synchronoizes the gradients computed
    from each tower, averages them and applies to the shared variables.

    See https://www.tensorflow.org/performance/benchmarks for details.
    """

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

    @staticmethod
    def setup_graph(model, input, ps_device, towers):
        """
        Args:
            model (ModelDesc):
            input (InputSource):
            ps_device (str):
            towers (list[int]):

        Returns:
            tf.Operation: the training op

            [Callback]: the callbacks to be added
        """
        callbacks = input.setup(model.get_inputs_desc())

        raw_devices = ['/gpu:{}'.format(k) for k in towers]
        if ps_device == 'gpu':
            devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        else:
            devices = [tf.train.replica_device_setter(
                worker_device=d, ps_device='/cpu:0', ps_tasks=1) for d in raw_devices]

        grad_list = MultiGPUTrainerBase.build_on_multi_tower(
            towers,
            lambda: MultiGPUTrainerBase._build_graph_get_grads(model, input),
            devices)
        MultiGPUTrainerBase._check_grad_list(grad_list)

        # debug tower performance (without update):
        # ops = [k[0] for k in grad_list[1]] + [k[0] for k in grad_list[0]]
        # self.train_op = tf.group(*ops)
        # return

        grads = SyncMultiGPUTrainerParameterServer._average_grads(grad_list)
        # grads = grad_list[0]

        opt = model.get_optimizer()
        if ps_device == 'cpu':
            with tf.device('/cpu:0'):
                train_op = opt.apply_gradients(grads, name='train_op')
        else:
            train_op = opt.apply_gradients(grads, name='train_op')
        return train_op, callbacks

    def _setup(self):
        self.train_op, cbs = SyncMultiGPUTrainerParameterServer.setup_graph(
            self.model, self._input_source, self._ps_device, self.config.tower)
        self.config.callbacks.extend(cbs)


def SyncMultiGPUTrainer(config):
    """
    Alias for ``SyncMultiGPUTrainerParameterServer(config, ps_device='gpu')``,
    as this is the most commonly used synchronous multigpu trainer (but may
    not be more efficient than the other).
    """
    return SyncMultiGPUTrainerParameterServer(config, ps_device='gpu')


class SyncMultiGPUTrainerReplicated(MultiGPUTrainerBase):
    """
    Data-parallel multi-GPU trainer where each GPU contains a replicate of the whole model.
    It will build one tower on each GPU under its own variable scope.
    Each gradient update is averaged across or GPUs through NCCL.

    See https://www.tensorflow.org/performance/benchmarks for details.
    """
    def __init__(self, config, gpu_prefetch=True):
        """
        Args:
            config, gpu_prefetch: same as in :class:`SyncMultiGPUTrainerParameterServer`
        """
        apply_prefetch_policy(config, gpu_prefetch)
        self._input_source = config.data
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

    @staticmethod
    def setup_graph(model, input, tower):
        """
        Args:
            model (ModelDesc):
            input (InputSource):
            tower (list[int]):

        Returns:
            tf.Operation: the training op

            [Callback]: the callbacks to be added
        """
        callbacks = input.setup(model.get_inputs_desc())

        raw_devices = ['/gpu:{}'.format(k) for k in tower]

        grad_list = MultiGPUTrainerBase.build_on_multi_tower(
            tower,
            lambda: MultiGPUTrainerBase._build_graph_get_grads(model, input),
            # use no variable scope for the first tower
            use_vs=[False] + [True] * (len(tower) - 1))
        grads = SyncMultiGPUTrainerReplicated._allreduce_grads(grad_list)

        train_ops = []
        opt = model.get_optimizer()
        for idx in range(len(tower)):
            with tf.device(raw_devices[idx]):
                grad_and_vars = [x[idx] for x in grads]
                # apply_gradients may create variables. Make them LOCAL_VARIABLES
                with override_to_local_variable(enable=idx > 0):
                    train_ops.append(opt.apply_gradients(
                        grad_and_vars, name='apply_grad_{}'.format(idx)))
        train_op = tf.group(*train_ops, name='train_op')
        cb = RunOp(
            SyncMultiGPUTrainerReplicated.get_post_init_ops,
            run_before=True, run_as_trigger=True, verbose=True)
        return train_op, callbacks + [cb]

    def _setup(self):
        self.train_op, cbs = SyncMultiGPUTrainerReplicated.setup_graph(
            self.model, self._input_source, self.config.tower)
        self.config.callbacks.extend(cbs)

# Adopt from https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/variable_mgr.py
    @staticmethod
    def get_post_init_ops():
        """
        Copy values of variables on GPU 0 to other GPUs.
        """
        # literally all variables, because it's better to sync optimizer-internal variables as well
        all_vars = tf.global_variables() + tf.local_variables()
        var_by_name = dict([(v.name, v) for v in all_vars])
        post_init_ops = []
        for v in all_vars:
            if not v.name.startswith('tower'):
                continue
            if v.name.startswith('tower0'):
                logger.warn("[SyncMultiGPUTrainerReplicated] variable "
                            "{} has prefix 'tower0', this is unexpected.".format(v.name))
                continue        # TODO some vars (EMA) may still startswith tower0
            # in this trainer, the master name doesn't have the towerx/ prefix
            split_name = v.name.split('/')
            prefix = split_name[0]
            realname = '/'.join(split_name[1:])
            if prefix in realname:
                logger.error("[SyncMultiGPUTrainerReplicated] variable "
                             "{} has its prefix {} appears multiple times in its name!".format(v.name, prefix))
            copy_from = var_by_name[realname]
            post_init_ops.append(v.assign(copy_from.read_value()))
        logger.info(
            "'sync_variables_from_main_tower' includes {} operations.".format(len(post_init_ops)))
        return tf.group(*post_init_ops, name='sync_variables_from_main_tower')


class AsyncMultiGPUTrainer(MultiGPUTrainerBase):
    """
    A data-parallel multi-GPU trainer. It builds one tower on each GPU with shared variable scope.
    Every tower computes the gradients and independently applies them to the
    variables, without synchronizing and averaging across towers.
    """

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

    @staticmethod
    def setup_graph(model, input, scale_gradient, tower):
        """
        Args:
            model (ModelDesc):
            input (InputSource):
            scale_gradient (bool):
            tower (list[int]):

        Returns:
            tf.Operation: the training op

            [Callback]: the callbacks to be added
        """
        callbacks = input.setup(model.get_inputs_desc())

        ps_device = 'cpu' if len(tower) >= 4 else 'gpu'

        if ps_device == 'gpu':
            raw_devices = ['/gpu:{}'.format(k) for k in tower]
            devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        else:
            devices = [tf.train.replica_device_setter(
                worker_device=d, ps_device='/cpu:0', ps_tasks=1) for d in raw_devices]
        grad_list = MultiGPUTrainerBase.build_on_multi_tower(
            tower,
            lambda: MultiGPUTrainerBase._build_graph_get_grads(model, input), devices)
        MultiGPUTrainerBase._check_grad_list(grad_list)

        if scale_gradient and len(tower) > 1:
            # pretend to average the grads, in order to make async and
            # sync have consistent effective learning rate
            gradproc = ScaleGradient(('.*', 1.0 / len(tower)), verbose=False)
            grad_list = [gradproc.process(gv) for gv in grad_list]
        # Ngpu x Nvar x 2

        train_ops = []
        opt = model.get_optimizer()
        with tf.name_scope('async_apply_gradients'):
            for i, grad_and_vars in enumerate(zip(*grad_list)):
                # Ngpu x 2
                v = grad_and_vars[0][1]
                with tf.device(v.device):
                    # will call apply_gradients (therefore gradproc) multiple times
                    train_ops.append(opt.apply_gradients(
                        grad_and_vars, name='apply_grad_{}'.format(i)))
            return tf.group(*train_ops, name='train_op'), callbacks

    def _setup(self):
        self.train_op, cbs = AsyncMultiGPUTrainer.setup_graph(
            self.model, self._input_source, self._scale_gradient, self.config.tower)
        self.config.callbacks.extend(cbs)
