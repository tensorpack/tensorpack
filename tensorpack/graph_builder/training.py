# -*- coding: utf-8 -*-
# File: training.py

import copy
import pprint
import re
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import six
import tensorflow as tf

from ..compat import tfv1
from ..tfutils.common import get_tf_version_tuple
from ..tfutils.gradproc import ScaleGradient
from ..tfutils.tower import TrainTowerContext
from ..utils import logger
from ..utils.develop import HIDE_DOC
from .utils import (
    GradientPacker, LeastLoadedDeviceSetter, aggregate_grads, allreduce_grads, allreduce_grads_hierarchical,
    merge_grad_list, override_to_local_variable, split_grad_list)

__all__ = ["DataParallelBuilder"]


@six.add_metaclass(ABCMeta)
class GraphBuilder(object):
    @abstractmethod
    def build(*args, **kwargs):
        pass


@contextmanager
def _maybe_reuse_vs(reuse):
    if reuse:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            yield
    else:
        yield


class DataParallelBuilder(GraphBuilder):
    def __init__(self, towers):
        """
        Args:
            towers(list[int]): list of GPU ids.
        """
        if len(towers) > 1:
            logger.info("[DataParallel] Training a model of {} towers.".format(len(towers)))
            if not tf.test.is_built_with_cuda():
                logger.error("[DataParallel] TensorFlow was not built with CUDA support!")

        self.towers = towers

    @staticmethod
    def _check_grad_list(grad_list):
        """
        Args:
            grad_list: list of list of tuples, shape is Ngpu x Nvar x 2
        """
        nvars = [len(k) for k in grad_list]

        def basename(x):
            return re.sub('tower[0-9]+/', '', x.op.name)

        if len(set(nvars)) != 1:
            names_per_gpu = [{basename(k[1]) for k in grad_and_vars} for grad_and_vars in grad_list]
            inters = copy.copy(names_per_gpu[0])
            for s in names_per_gpu:
                inters &= s
            for s in names_per_gpu:
                s -= inters
            logger.error("Unique trainable variables on towers: " + pprint.pformat(names_per_gpu))
            raise ValueError("Number of gradients from each tower is different! " + str(nvars))

    @staticmethod
    def call_for_each_tower(
            towers, func, devices=None, use_vs=None):
        """
        Run `func` on all GPUs (towers) and return the results.

        Args:
            towers (list[int]): a list of GPU id.
            func: a lambda to be called inside each tower
            devices: a list of devices to be used. By default will use '/gpu:{tower}'
            use_vs (list[bool]): list of use_vs to passed to TowerContext

        Returns:
            List of outputs of ``func``, evaluated on each tower.
        """

        ret = []
        if devices is not None:
            assert len(devices) == len(towers)
        if use_vs is not None:
            assert len(use_vs) == len(towers)

        tower_names = ['tower{}'.format(idx) for idx in range(len(towers))]

        for idx, t in enumerate(towers):
            device = devices[idx] if devices is not None else '/gpu:{}'.format(t)
            usevs = use_vs[idx] if use_vs is not None else False
            reuse = not usevs and idx > 0
            with tfv1.device(device), _maybe_reuse_vs(reuse), TrainTowerContext(
                    tower_names[idx],
                    vs_name=tower_names[idx] if usevs else '',
                    index=idx, total=len(towers)):
                if len(str(device)) < 10:   # a device function doesn't have good string description
                    logger.info("Building graph for training tower {} on device {} ...".format(idx, device))
                else:
                    logger.info("Building graph for training tower {} ...".format(idx))

                # When use_vs is True, use LOCAL_VARIABLES,
                # so these duplicated variables won't be saved by default.
                with override_to_local_variable(enable=usevs):
                    ret.append(func())
        return ret

    @staticmethod
    @HIDE_DOC
    def build_on_towers(*args, **kwargs):
        return DataParallelBuilder.call_for_each_tower(*args, **kwargs)


class SyncMultiGPUParameterServerBuilder(DataParallelBuilder):
    """
    Data-parallel training in 'ParameterServer' mode.
    It builds one tower on each GPU with
    shared variable scope. It synchronizes the gradients computed
    from each tower, averages them and applies to the shared variables.

    It is an equivalent of ``--variable_update=parameter_server`` in
    `tensorflow/benchmarks <https://github.com/tensorflow/benchmarks>`_.
    """
    def __init__(self, towers, ps_device):
        """
        Args:
            towers(list[int]): list of GPU id
            ps_device (str): either 'gpu' or 'cpu', where variables are stored.
        """
        super(SyncMultiGPUParameterServerBuilder, self).__init__(towers)
        assert ps_device in ['cpu', 'gpu']
        self.ps_device = ps_device

    def call_for_each_tower(self, tower_fn):
        """
        Call the function `tower_fn` under :class:`TowerContext` for each tower.

        Returns:
            a list, contains the return values of `tower_fn` on each tower.
        """
        raw_devices = ['/gpu:{}'.format(k) for k in self.towers]
        if self.ps_device == 'gpu':
            devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        else:
            devices = [tf.train.replica_device_setter(
                worker_device=d, ps_device='/cpu:0', ps_tasks=1) for d in raw_devices]

        return DataParallelBuilder.build_on_towers(self.towers, tower_fn, devices)

    def build(self, grad_list, get_opt_fn):
        """
        Reduce the gradients, apply them with the optimizer,
        and set self.grads to a list of (g, v), containing the averaged gradients.

        Args:
            grad_list ([[(grad, var), ...], ...]): #GPU lists to be reduced. Each is the gradients computed on each GPU.
            get_opt_fn (-> tf.train.Optimizer): callable which returns an optimizer

        Returns:
            tf.Operation: the training op
        """
        assert len(grad_list) == len(self.towers)
        DataParallelBuilder._check_grad_list(grad_list)

        # debug tower performance (without update):
        # ops = [k[0] for k in grad_list[1]] + [k[0] for k in grad_list[0]]
        # self.train_op = tf.group(*ops)
        # return

        self.grads = aggregate_grads(grad_list, colocation=True)
        # grads = grad_list[0]

        opt = get_opt_fn()
        if self.ps_device == 'cpu':
            with tf.device('/cpu:0'):
                train_op = opt.apply_gradients(self.grads, name='train_op')
        else:
            train_op = opt.apply_gradients(self.grads, name='train_op')
        return train_op


class SyncMultiGPUReplicatedBuilder(DataParallelBuilder):
    """
    Data-parallel training in "replicated" mode,
    where each GPU contains a replicate of the whole model.
    It will build one tower on each GPU under its own variable scope.
    Each gradient update is averaged or summed across or GPUs through NCCL.

    It is an equivalent of ``--variable_update=replicated`` in
    `tensorflow/benchmarks <https://github.com/tensorflow/benchmarks>`_.
    """

    def __init__(self, towers, average, mode):
        super(SyncMultiGPUReplicatedBuilder, self).__init__(towers)
        self._average = average
        assert mode in ['nccl', 'cpu', 'hierarchical'], mode
        self._mode = mode

        if self._mode == 'hierarchical' and len(towers) != 8:
            logger.warn("mode='hierarchical' require >= 8 GPUs. Fallback to mode='nccl'.")
            self._mode = 'nccl'

    def call_for_each_tower(self, tower_fn):
        """
        Call the function `tower_fn` under :class:`TowerContext` for each tower.

        Returns:
            a list, contains the return values of `tower_fn` on each tower.
        """
        # if tower_fn returns [(grad, var), ...], this returns #GPU x #VAR x 2
        return DataParallelBuilder.build_on_towers(
            self.towers,
            tower_fn,
            # use no variable scope for the first tower
            use_vs=[False] + [True] * (len(self.towers) - 1))

    def build(self, grad_list, get_opt_fn):
        """
        Reduce the gradients, apply them with the optimizer,
        and set self.grads to #GPU number of lists of (g, v), containing the all-reduced gradients on each device.

        Args:
            grad_list ([[(grad, var), ...], ...]): #GPU lists to be reduced. Each is the gradients computed on each GPU.
            get_opt_fn (-> tf.train.Optimizer): callable which returns an optimizer

        Returns:
            (tf.Operation, tf.Operation)

            1. the training op.

            2. the op which sync variables from GPU 0 to other GPUs.
                It has to be run before the training has started.
                And you can optionally run it later to sync non-trainable variables.
        """
        assert len(grad_list) == len(self.towers)
        raw_devices = ['/gpu:{}'.format(k) for k in self.towers]

        DataParallelBuilder._check_grad_list(grad_list)

        dtypes = {x[0].dtype.base_dtype for x in grad_list[0]}
        dtypes_nccl_supported = [tf.float32, tf.float64]
        if get_tf_version_tuple() >= (1, 8):
            dtypes_nccl_supported.append(tf.float16)
        valid_for_nccl = all(k in dtypes_nccl_supported for k in dtypes)
        if self._mode == 'nccl' and not valid_for_nccl:
            logger.warn("Cannot use mode='nccl' because some gradients have unsupported types. Fallback to mode='cpu'")
            self._mode = 'cpu'

        if self._mode in ['nccl', 'hierarchical']:
            all_grads, all_vars = split_grad_list(grad_list)
            # use allreduce from tf-benchmarks
            # from .batch_allreduce import AllReduceSpecAlgorithm
            # algo = AllReduceSpecAlgorithm('nccl', list(range(8)), 0, 10)
            # all_grads, warmup_ops = algo.batch_all_reduce(all_grads, 1, True, False)
            # print("WARMUP OPS", warmup_ops)

            if self._mode == 'nccl':
                all_grads = allreduce_grads(all_grads, average=self._average)  # #gpu x #param
            else:
                packer = GradientPacker(len(raw_devices))
                succ = packer.compute_strategy(all_grads[0])
                if succ:
                    packed_grads = packer.pack_all(all_grads, raw_devices)
                    packed_grads_aggr = allreduce_grads_hierarchical(
                        packed_grads, raw_devices, average=self._average)
                    all_grads = packer.unpack_all(packed_grads_aggr, raw_devices)
                else:
                    all_grads = allreduce_grads_hierarchical(all_grads, raw_devices, average=self._average)

            self.grads = merge_grad_list(all_grads, all_vars)
        elif self._mode == 'cpu':
            agg_grad_and_vars = aggregate_grads(
                grad_list, colocation=False,
                devices=['/cpu:0'], average=self._average)    # #param x 2
            self.grads = []  # #gpu x #param x 2
            for grad_and_vars in grad_list:   # grad_and_vars: #paramx2
                # take v from each tower, and g from average.
                self.grads.append(
                    [(g, v) for (_, v), (g, _) in zip(grad_and_vars, agg_grad_and_vars)])

        train_ops = []
        opt = get_opt_fn()
        with tf.name_scope('apply_gradients'):
            for idx, grad_and_vars in enumerate(self.grads):
                with tf.device(raw_devices[idx]):
                    # apply_gradients may create variables. Make them LOCAL_VARIABLES
                    with override_to_local_variable(enable=idx > 0):
                        train_ops.append(opt.apply_gradients(
                            grad_and_vars, name='apply_grad_{}'.format(idx)))
        train_op = tf.group(*train_ops, name='train_op')

        if len(self.towers) > 1:
            with tf.name_scope('sync_variables'):
                post_init_op = SyncMultiGPUReplicatedBuilder.get_post_init_ops()
        else:
            post_init_op = None
        return train_op, post_init_op

# Adopt from https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/variable_mgr.py
    @staticmethod
    def get_post_init_ops():
        """
        Copy values of variables on GPU 0 to other GPUs.
        """
        # literally all variables, because it's better to sync optimizer-internal variables as well
        all_vars = tf.global_variables() + tf.local_variables()
        var_by_name = {v.name: v for v in all_vars}
        trainable_names = {x.name for x in tf.trainable_variables()}
        post_init_ops = []

        def log_failure(name, reason):
            logger.warn("[ReplicatedTrainer] Do not know how to sync variable '{}' across GPUs. "
                        "Reason: {} ".format(name, reason))
            assert name not in trainable_names, \
                "The aforementioned variable is trainable, so this is probably a fatal error."
            logger.warn(
                "[ReplicatedTrainer] This variable is non-trainable. "
                "Ignore this warning if you know it's OK to leave it out-of-sync.")

        for v in all_vars:
            if not v.name.startswith('tower'):
                continue
            if v.name.startswith('tower0'):
                # in this trainer, the master name doesn't have the towerx/ prefix
                log_failure(v.name, "Name should not have prefix 'tower0' in this trainer!")
                continue        # TODO some vars (EMA) may still startswith tower0

            split_name = v.name.split('/')
            prefix = split_name[0]
            realname = '/'.join(split_name[1:])
            if prefix in realname:
                log_failure(v.name, "Prefix {} appears multiple times in its name!".format(prefix))
                continue
            copy_from = var_by_name.get(realname)
            if copy_from is not None:
                post_init_ops.append(v.assign(copy_from.read_value()))
            else:
                log_failure(v.name, "Cannot find {} in the graph!".format(realname))
        logger.info(
            "'sync_variables_from_main_tower' includes {} operations.".format(len(post_init_ops)))
        return tf.group(*post_init_ops, name='sync_variables_from_main_tower')


class AsyncMultiGPUBuilder(DataParallelBuilder):
    """
    Data-parallel training with async update.
    It builds one tower on each GPU with shared variable scope.
    Every tower computes the gradients and independently applies them to the
    variables, without synchronizing and averaging across towers.
    """

    def __init__(self, towers, scale_gradient=True):
        """
        Args:
            towers(list[int]): list of GPU ids.
            scale_gradient (bool): if True, will scale each gradient by ``1.0/nr_gpu``.
        """
        super(AsyncMultiGPUBuilder, self).__init__(towers)
        self._scale_gradient = scale_gradient

    def call_for_each_tower(self, tower_fn):
        """
        Call the function `tower_fn` under :class:`TowerContext` for each tower.

        Returns:
            a list, contains the return values of `tower_fn` on each tower.
        """
        ps_device = 'cpu' if len(self.towers) >= 4 else 'gpu'

        raw_devices = ['/gpu:{}'.format(k) for k in self.towers]
        if ps_device == 'gpu':
            devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        else:
            devices = [tf.train.replica_device_setter(
                worker_device=d, ps_device='/cpu:0', ps_tasks=1) for d in raw_devices]

        return DataParallelBuilder.build_on_towers(self.towers, tower_fn, devices)

    def build(self, grad_list, get_opt_fn):
        """
        Args:
            grad_list ([[(grad, var), ...], ...]): #GPU lists to be reduced. Each is the gradients computed on each GPU.
            get_opt_fn (-> tf.train.Optimizer): callable which returns an optimizer

        Returns:
            tf.Operation: the training op
        """
        assert len(grad_list) == len(self.towers)
        DataParallelBuilder._check_grad_list(grad_list)

        if self._scale_gradient and len(self.towers) > 1:
            # pretend to average the grads, in order to make async and
            # sync have consistent effective learning rate
            gradproc = ScaleGradient(('.*', 1.0 / len(self.towers)), verbose=False)
            grad_list = [gradproc.process(gv) for gv in grad_list]
        # Ngpu x Nvar x 2

        train_ops = []
        opt = get_opt_fn()
        with tf.name_scope('async_apply_gradients'):
            for i, grad_and_vars in enumerate(zip(*grad_list)):
                # Ngpu x 2
                v = grad_and_vars[0][1]
                with tf.device(v.device):
                    # will call apply_gradients (therefore gradproc) multiple times
                    train_ops.append(opt.apply_gradients(
                        grad_and_vars, name='apply_grad_{}'.format(i)))
        return tf.group(*train_ops, name='train_op')
