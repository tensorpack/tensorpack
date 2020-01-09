# -*- coding: utf-8 -*-
# File: trainers.py

import multiprocessing as mp
import os
import sys
import tensorflow as tf

from ..callbacks import CallbackFactory, RunOp
from ..graph_builder.distributed import DistributedParameterServerBuilder, DistributedReplicatedBuilder
from ..graph_builder.training import (
    AsyncMultiGPUBuilder, SyncMultiGPUParameterServerBuilder, SyncMultiGPUReplicatedBuilder)
from ..graph_builder.utils import override_to_local_variable
from ..input_source import FeedfreeInput, QueueInput
from ..tfutils import get_global_step_var
from ..tfutils.distributed import get_distributed_session_creator
from ..tfutils.sesscreate import NewSessionCreator
from ..tfutils.tower import TrainTowerContext
from ..utils import logger
from ..utils.argtools import map_arg
from ..utils.develop import HIDE_DOC, deprecated
from .tower import SingleCostTrainer

__all__ = ['NoOpTrainer', 'SimpleTrainer',
           'QueueInputTrainer',
           'SyncMultiGPUTrainer',
           'SyncMultiGPUTrainerReplicated',
           'SyncMultiGPUTrainerParameterServer',
           'AsyncMultiGPUTrainer',
           'DistributedTrainerParameterServer',
           'DistributedTrainerReplicated',
           'HorovodTrainer', 'BytePSTrainer']


def _int_to_range(x):
    if isinstance(x, int):
        assert x > 0, "Argument cannot be {}!".format(x)
        return list(range(x))
    return x


class SimpleTrainer(SingleCostTrainer):
    """
    Single-GPU single-cost single-tower trainer.
    """
    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        logger.info("Building graph for a single training tower ...")
        with TrainTowerContext(''):
            grads = self._make_get_grad_fn(input, get_cost_fn, get_opt_fn)()
            opt = get_opt_fn()
            self.train_op = opt.apply_gradients(grads, name='train_op')
        return []


class NoOpTrainer(SimpleTrainer):
    """
    A special trainer that builds the graph (if given a tower function)
    and does nothing in each step.
    It is used to only run the callbacks.

    Note that `steps_per_epoch` and `max_epochs` are still valid options.
    """
    def run_step(self):
        self.hooked_sess.run([])


# Only exists for type check & back-compatibility
class QueueInputTrainer(SimpleTrainer):
    @deprecated("SimpleTrainer is sufficient!", "2019-12-31")
    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        assert isinstance(input, QueueInput), input
        return super(QueueInputTrainer, self)._setup_graph(input, get_cost_fn, get_opt_fn)


class SyncMultiGPUTrainerParameterServer(SingleCostTrainer):

    __doc__ = SyncMultiGPUParameterServerBuilder.__doc__ + """

    Attributes:
        devices (list[int]): List of GPU ids.

    """

    @map_arg(gpus=_int_to_range)
    def __init__(self, gpus, ps_device=None):
        """
        Args:
            gpus ([int]): list of GPU ids.
            ps_device: either 'gpu' or 'cpu', where variables are stored.
                The default value is subject to change.
        """
        self.devices = gpus
        if ps_device is None:
            ps_device = 'gpu' if len(gpus) <= 2 else 'cpu'
        self._builder = SyncMultiGPUParameterServerBuilder(gpus, ps_device)
        super(SyncMultiGPUTrainerParameterServer, self).__init__()

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        if len(self.devices) > 1:
            assert isinstance(input, FeedfreeInput), input
        tower_fn = self._make_get_grad_fn(input, get_cost_fn, get_opt_fn)
        grad_list = self._builder.call_for_each_tower(tower_fn)
        self.train_op = self._builder.build(grad_list, get_opt_fn)
        return []


def SyncMultiGPUTrainer(gpus):
    """
    Return a default multi-GPU trainer, if you don't care about the details.
    It may not be the most efficient one for your task.

    Args:
        gpus (list[int]): list of GPU ids.
    """
    return SyncMultiGPUTrainerParameterServer(gpus, ps_device='cpu')


class AsyncMultiGPUTrainer(SingleCostTrainer):

    __doc__ = AsyncMultiGPUBuilder.__doc__ + """

    Attributes:
        devices (list[int]): List of GPU ids.

    """

    @map_arg(gpus=_int_to_range)
    def __init__(self, gpus, scale_gradient=True):
        """
        Args:
            gpus ([int]): list of GPU ids.
            scale_gradient (bool): if True, will scale each gradient by ``1.0/nr_gpu``.
        """
        self.devices = gpus
        self._builder = AsyncMultiGPUBuilder(gpus, scale_gradient)
        super(AsyncMultiGPUTrainer, self).__init__()

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        if len(self.devices) > 1:
            assert isinstance(input, FeedfreeInput), input
        tower_fn = self._make_get_grad_fn(input, get_cost_fn, get_opt_fn)
        grad_list = self._builder.call_for_each_tower(tower_fn)
        self.train_op = self._builder.build(grad_list, get_opt_fn)
        return []


class SyncMultiGPUTrainerReplicated(SingleCostTrainer):

    __doc__ = SyncMultiGPUReplicatedBuilder.__doc__ + """

    Attributes:
        devices (list[int]): List of GPU ids.

        BROADCAST_EVERY_EPOCH (bool):
            Whether to broadcast the variables every epoch.
            Theoretically this is a no-op (because the variables
            are supposed to be in-sync).
            But this cheap operation may help prevent
            certain numerical issues in practice.
            Note that in cases such as BatchNorm, the variables may not be in sync.
    """

    @map_arg(gpus=_int_to_range)
    def __init__(self, gpus, average=True, mode=None):
        """
        Args:
            gpus (int or [int]): list of GPU ids.
            average (bool): whether to average or sum gradients.
            mode (str or None): Gradient aggregation mode.
                Supported values: ['nccl', 'hierarchical', 'cpu'].
                Default to pick automatically by heuristics.
                These modes may have slight (within 5%) differences in speed.
                "hierarchical" mode was designed for DGX-like 8GPU machines.
        """
        self.devices = gpus

        if mode is None:
            mode = 'hierarchical' if len(gpus) == 8 else 'nccl'
        mode = mode.lower()

        self._builder = SyncMultiGPUReplicatedBuilder(gpus, average, mode)
        self.BROADCAST_EVERY_EPOCH = True

        super(SyncMultiGPUTrainerReplicated, self).__init__()

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        if len(self.devices) > 1:
            assert isinstance(input, FeedfreeInput), input
        tower_fn = self._make_get_grad_fn(input, get_cost_fn, get_opt_fn)
        grad_list = self._builder.call_for_each_tower(tower_fn)
        self.train_op, post_init_op = self._builder.build(grad_list, get_opt_fn)

        if post_init_op is not None:
            cb = RunOp(
                post_init_op,
                run_before=True,
                run_as_trigger=self.BROADCAST_EVERY_EPOCH,
                verbose=True)
            cb.name_scope = "SyncVariables"
            return [cb]
        else:
            return []


class DistributedTrainerBase(SingleCostTrainer):

    devices = None

    def __init__(self, gpus, server):
        super(DistributedTrainerBase, self).__init__()
        self.devices = gpus
        self.server = server
        self.job_name = server.server_def.job_name
        logger.info("Distributed training on cluster:\n" + str(server.server_def.cluster))

    def join(self):
        logger.info("Calling server.join() on {}:{}".format(self.job_name, self.server.server_def.task_index))
        logger.info("Kill me with 'kill {}'".format(os.getpid()))
        self.server.join()  # this function will never return tensorflow#4713
        raise RuntimeError("This is a bug. Server.join() for should never return!")

    @HIDE_DOC
    def initialize(self, session_creator, session_init):
        if not isinstance(session_creator, NewSessionCreator) or \
                session_creator.user_provided_config:
            raise ValueError(
                "You are not allowed to set session_creator or session_config for distributed training! "
                "To use a custom session config, pass it to tf.train.Server.")
        super(DistributedTrainerBase, self).initialize(
            get_distributed_session_creator(self.server), session_init)


# This is slow. deprecated in favor of horovod
class DistributedTrainerParameterServer(DistributedTrainerBase):

    __doc__ = DistributedParameterServerBuilder.__doc__

    @map_arg(gpus=_int_to_range)
    def __init__(self, gpus, server, caching_device='cpu'):
        """
        Args:
            gpus ([int]): list of GPU ids.
            server (tf.train.Server): the server with ps and workers.
            caching_device (str): either 'cpu' or 'gpu'. The device to cache variables copied from PS
        """
        super(DistributedTrainerParameterServer, self).__init__(gpus, server)
        assert self.job_name in ['ps', 'worker'], self.job_name
        if self.job_name == 'ps':
            self.join()

        self._builder = DistributedParameterServerBuilder(gpus, server, caching_device)
        self.is_chief = self._builder.is_chief

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        assert isinstance(input, FeedfreeInput), input
        self.train_op = self._builder.build(
            self._make_get_grad_fn(input, get_cost_fn, get_opt_fn), get_opt_fn)
        return []


# This is slow. deprecated in favor of horovod
class DistributedTrainerReplicated(DistributedTrainerBase):

    __doc__ = DistributedReplicatedBuilder.__doc__

    @map_arg(gpus=_int_to_range)
    def __init__(self, gpus, server):
        """
        Args:
            gpus (list[int]): list of GPU ids.
            server (tf.train.Server): the server with ps and workers.
        """
        super(DistributedTrainerReplicated, self).__init__(gpus, server)
        assert self.job_name in ['ps', 'worker'], self.job_name
        if self.job_name == 'ps':
            self.join()

        self._builder = DistributedReplicatedBuilder(gpus, server)
        self.is_chief = self._builder.is_chief

    def _setup_input(self, input_signature, input):
        with override_to_local_variable():
            get_global_step_var()  # gs should be local
            # input source may create variables (queue size summary)
            # TODO This is not good because we don't know from here
            # whether something should be global or local. We now assume
            # they should be local.
            assert not input.setup_done()
            return input.setup(input_signature)

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        assert isinstance(input, FeedfreeInput), input
        self.train_op, initial_sync_op, model_sync_op = self._builder.build(
            self._make_get_grad_fn(input, get_cost_fn, get_opt_fn), get_opt_fn)

        callbacks = []
        # Initial syncing vars from PS
        cb = RunOp(lambda: initial_sync_op,
                   run_before=True, run_as_trigger=False, verbose=True)
        cb.chief_only = False
        callbacks.append(cb)

        # Sync model_variables to PS, only chief needs to do this
        if model_sync_op:
            cb = RunOp(lambda: model_sync_op,
                       run_before=False, run_as_trigger=True, verbose=True)
            logger.warn("For efficiency, local MODEL_VARIABLES are only synced to PS once "
                        "every epoch. Be careful if you save the model more frequently than this.")
            callbacks.append(cb)
        return callbacks

    @property
    def _main_tower_vs_name(self):
        return "tower0"


class HorovodTrainer(SingleCostTrainer):
    """
    Horovod trainer, support both multi-GPU and distributed training.

    To use for multi-GPU training:

    .. code-block:: bash

        # First, change trainer to HorovodTrainer(), then
        CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO mpirun -np 4 --output-filename mylog python train.py

    To use for distributed training:

    .. code-block:: bash

        # First, change trainer to HorovodTrainer(), then
        mpirun -np 8 -H server1:4,server2:4  \\
            -bind-to none -map-by slot \\
            --output-filename mylog -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \\
            python train.py
        # Add other environment variables you need by -x, e.g. PYTHONPATH, PATH.
        # If using all GPUs, you can always skip the `CUDA_VISIBLE_DEVICES` option.
        # There are other MPI options that can potentially improve performance especially on special hardwares.

    Horovod can also be launched without MPI. See
    `its documentation <https://github.com/horovod/horovod#running-horovod>`_
    for more details.

    Note:
        1. To reach the maximum speed in your system, there are many options to tune
           for Horovod installation and in the MPI command line.
           See Horovod docs for details.

        2. Due to a TF bug (#8136), you must not initialize CUDA context before the trainer starts training.
           Therefore TF functions like `is_gpu_available()` or `list_local_devices()`
           must be avoided.
           You can, however, use `tf.config.experimental.list_physical_devices('GPU')`, introduced in TF 1.14.

        3. Horovod supports both MPI and gloo. There are a few drawbacks of the MPI backend:

            + MPI does not like `fork()`. If your code (e.g. dataflow) contains multiprocessing, it may cause problems.
            + MPI sometimes fails to kill all processes in the end. Be sure to check it afterwards.

        4. Keep in mind that there is one process running the script per GPU, therefore:

           + Make sure your InputSource has reasonable randomness.

           + If your data processing is heavy, doing it in a single dedicated process might be
             a better choice than doing them repeatedly in each process.

           + You need to make sure log directories in each process won't conflict.
             You can set it only for the chief process, or set a different one for each process.

           + Callbacks have an option to be run only in the chief process, or in all processes.
             See :meth:`Callback.set_chief_only()`. Most callbacks have a reasonable
             default already, but certain callbacks may need your customization.
             Report an issue if you find any bad defaults.

           + You can use Horovod API such as `hvd.rank()` to know which process you are and choose
             different code path. Chief process has rank 0.

        5. Due to these caveats, see
           `ResNet-Horovod <https://github.com/tensorpack/benchmarks/tree/master/ResNet-Horovod>`_
           for a full example which has handled these common issues.
           This example can train ImageNet in roughly an hour following the paper's setup.

    Attributes:
        BROADCAST_EVERY_EPOCH (bool):
            Whether to broadcast the variables every epoch.
            Theoretically this is a no-op (because the variables
            are supposed to be in-sync).
            But this cheap operation may help prevent certain numerical issues in practice.
            Note that in cases such as BatchNorm, the variables may not be in sync.
    """

    def __init__(self, average=True, compression=None):
        """
        Args:
            average (bool): whether to average or sum the gradients across processes.
            compression: `hvd.Compression.fp16` or `hvd.Compression.none`
        """
        if 'pyarrow' in sys.modules:
            logger.warn("Horovod and pyarrow may conflict due to pyarrow bugs. "
                        "Uninstall pyarrow and use msgpack instead.")
        # lazy import
        import horovod.tensorflow as hvd
        import horovod
        hvd_version = tuple(map(int, horovod.__version__.split('.')[:3]))
        self.hvd = hvd

        hvd.init()
        self.is_chief = hvd.rank() == 0
        self._local_rank = hvd.local_rank()
        self._rank = hvd.rank()
        self._average = average
        self._compression = compression
        self._has_compression = hvd_version >= (0, 15, 0)
        logger.info("[HorovodTrainer] local rank={}".format(self._local_rank))
        super(HorovodTrainer, self).__init__()

        self.BROADCAST_EVERY_EPOCH = True

    def mpi_enabled(self):
        """
        Returns:
            bool: whether hvd is currently running under MPI
        """
        try:
            return self.hvd.mpi_enabled()
        except AttributeError:
            return False

    def allreduce(self, grads):
        if self.hvd.size() == 1:
            return grads
        # copied from https://github.com/uber/horovod/blob/master/horovod/tensorflow/__init__.py
        averaged_gradients = []
        with tf.name_scope("AllReduce"):
            for grad, var in grads:
                if grad is not None:
                    if self._compression is not None and self._has_compression:
                        avg_grad = self.hvd.allreduce(grad, average=self._average, compression=self._compression)
                    else:
                        avg_grad = self.hvd.allreduce(grad, average=self._average)
                    averaged_gradients.append((avg_grad, var))
                else:
                    averaged_gradients.append((None, var))
        return averaged_gradients

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        with TrainTowerContext(''):
            grads = self._make_get_grad_fn(input, get_cost_fn, get_opt_fn)()
            grads = self.allreduce(grads)

            opt = get_opt_fn()
            self.train_op = opt.apply_gradients(grads, name='train_op')

        cb = CallbackFactory(
            before_train=self.broadcast,
            trigger=self.broadcast if self.BROADCAST_EVERY_EPOCH else None
        ).set_chief_only(False)
        return [cb]

    def broadcast(self, _):
        logger.info("Running broadcast ...")
        # the op will be created in initialize()
        self.sess.run(self._broadcast_op)

    @HIDE_DOC
    def initialize(self, session_creator, session_init):
        # broadcast_op should be the last setup_graph: it needs to be created
        # "right before" the graph is finalized,
        # because it needs to capture all the variables (which may be created by callbacks).
        self._broadcast_op = self.hvd.broadcast_global_variables(0)

        # it's important that our NewSessionCreator does not finalize the graph
        if not isinstance(session_creator, NewSessionCreator):
            raise ValueError(
                "session_creator has to be `NewSessionCreator` for horovod/byteps training! ")
        # NOTE It will fail if GPU was already detected before initializing the session
        # https://github.com/tensorflow/tensorflow/issues/8136
        session_creator.config.gpu_options.visible_device_list = str(self._local_rank)
        try:
            session_creator.config.inter_op_parallelism_threads = mp.cpu_count() // self.hvd.local_size()
        except AttributeError:  # old horovod does not have local_size
            pass
        super(HorovodTrainer, self).initialize(session_creator, session_init)

        # This broadcast belongs to the "intialize" stage
        # It should not be delayed to the "before_train" stage.
        # TODO:
        # 1. a allgather helper to concat strings
        # 2. check variables on each rank match each other, print warnings, and broadcast the common set.
        if self.is_chief:
            logger.info("Broadcasting initialized variables ...")
        else:
            logger.info("Rank {} waiting for initialization broadcasting ...".format(self._rank))
        self.sess.run(self._broadcast_op)


class BytePSTrainer(HorovodTrainer):
    """
    BytePS trainer. Supports both multi-GPU and distributed training.
    It achieves better scalability than horovod in distributed training, if the model is communication
    intensive and you have properly set up the machines following its
    `best practices <https://github.com/bytedance/byteps/blob/master/docs/best-practice.md>`_
    which requires a few extra bandwidth servers than horovod.

    To use it, switch the trainer, and refer to BytePS documentation on how to
    launch server/scheduler/workers.

    Attributes:
        hvd (module): the byteps module that contains horovod-compatible APIs
            like `rank(),size()`.
            This attribute exists so that downstream code that uses these APIs
            does not need to worry about which library is being used under the hood.
    """
    def __init__(self, average=True):
        """
        Args:
            average (bool): whether to average or sum the gradients across processes.
        """
        import byteps.tensorflow as bps
        self.hvd = bps  # BytePS has the same interface as Horovod
        self.hvd.allreduce = bps.push_pull  # https://github.com/bytedance/byteps/issues/8
        assert os.environ.get("DMLC_ROLE", None) == "worker"
        assert "DMLC_WORKER_ID" in os.environ and "DMLC_NUM_WORKER" in os.environ
        bps.init()
        self.is_chief = bps.rank() == 0

        self._local_rank = bps.local_rank()
        self._rank = bps.rank()
        self._average = average

        self._compression = None
        self._has_compression = False
        logger.info("[BytePSTrainer] local rank={}".format(self._local_rank))
        SingleCostTrainer.__init__(self)

    def mpi_enabled(self):
        """
        Returns:
            bool: whether hvd is currently running under MPI
        """
        return False
