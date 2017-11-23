#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trainers.py

import os
import tensorflow as tf

from ..callbacks import RunOp
from ..tfutils.sesscreate import NewSessionCreator

from ..utils import logger
from ..utils.argtools import map_arg
from ..utils.develop import HIDE_DOC
from ..tfutils import get_global_step_var
from ..tfutils.distributed import get_distributed_session_creator
from ..tfutils.tower import TowerContext
from ..input_source import QueueInput

from ..graph_builder.training import (
    SyncMultiGPUParameterServerBuilder,
    SyncMultiGPUReplicatedBuilder,
    AsyncMultiGPUBuilder)
from ..graph_builder.distributed import DistributedReplicatedBuilder
from ..graph_builder.utils import override_to_local_variable

from .tower import SingleCostTrainer

__all__ = ['SimpleTrainer',
           'QueueInputTrainer',
           'SyncMultiGPUTrainer',
           'SyncMultiGPUTrainerReplicated',
           'SyncMultiGPUTrainerParameterServer',
           'AsyncMultiGPUTrainer',
           'DistributedTrainerReplicated',
           'HorovodTrainer']


def _int_to_range(x):
    if isinstance(x, int):
        assert x > 0, x
        return list(range(x))
    return x


class SimpleTrainer(SingleCostTrainer):
    """
    Single-GPU single-cost single-tower trainer.
    """
    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        with TowerContext('', is_training=True):
            grads = self._make_get_grad_fn(input, get_cost_fn, get_opt_fn)()
            opt = get_opt_fn()
            self.train_op = opt.apply_gradients(grads, name='min_op')
        return []


# Only works for type check
class QueueInputTrainer(SimpleTrainer):
    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        assert isinstance(input, QueueInput)
        return super(QueueInputTrainer, self)._setup_graph(input, get_cost_fn, get_opt_fn)


class SyncMultiGPUTrainerParameterServer(SingleCostTrainer):

    __doc__ = SyncMultiGPUParameterServerBuilder.__doc__

    @map_arg(gpus=_int_to_range)
    def __init__(self, gpus, ps_device='gpu'):
        """
        Args:
            gpus ([int]): list of GPU ids.
            ps_device: either 'gpu' or 'cpu', where variables are stored.  Setting to 'cpu' might help when #gpu>=4
        """
        self._builder = SyncMultiGPUParameterServerBuilder(gpus, ps_device)
        super(SyncMultiGPUTrainerParameterServer, self).__init__()

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        self.train_op = self._builder.build(
            self._make_get_grad_fn(input, get_cost_fn, get_opt_fn), get_opt_fn)
        return []


def SyncMultiGPUTrainer(gpus):
    """
    Return a default multi-GPU trainer, if you don't care about the details.
    It may not be the most efficient one for your task.

    Args:
        gpus (list[int]): list of GPU ids.
    """
    return SyncMultiGPUTrainerParameterServer(gpus, ps_device='gpu')


class AsyncMultiGPUTrainer(SingleCostTrainer):

    __doc__ = AsyncMultiGPUBuilder.__doc__

    @map_arg(gpus=_int_to_range)
    def __init__(self, gpus, scale_gradient=True):
        """
        Args:
            gpus ([int]): list of GPU ids.
            scale_gradient (bool): if True, will scale each gradient by ``1.0/nr_gpu``.
        """
        self._builder = AsyncMultiGPUBuilder(gpus, scale_gradient)
        super(AsyncMultiGPUTrainer, self).__init__()

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        self.train_op = self._builder.build(
            self._make_get_grad_fn(input, get_cost_fn, get_opt_fn), get_opt_fn)
        return []


class SyncMultiGPUTrainerReplicated(SingleCostTrainer):

    __doc__ = SyncMultiGPUReplicatedBuilder.__doc__

    @map_arg(gpus=_int_to_range)
    def __init__(self, gpus):
        """
        Args:
            gpus ([int]): list of GPU ids.
        """
        self._builder = SyncMultiGPUReplicatedBuilder(gpus)
        super(SyncMultiGPUTrainerReplicated, self).__init__()

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        self.train_op, post_init_op = self._builder.build(
            self._make_get_grad_fn(input, get_cost_fn, get_opt_fn), get_opt_fn)

        cb = RunOp(
            post_init_op,
            run_before=True, run_as_trigger=True, verbose=True)
        return [cb]


class DistributedTrainerReplicated(SingleCostTrainer):

    __doc__ = DistributedReplicatedBuilder.__doc__

    @map_arg(gpus=_int_to_range)
    def __init__(self, gpus, server):
        """
        Args:
            gpus (list[int]): list of GPU ids.
            server (tf.train.Server): the server with ps and workers.
        """
        self.server = server
        self.job_name = server.server_def.job_name
        assert self.job_name in ['ps', 'worker'], self.job_name

        if self.job_name == 'worker':
            # ps doesn't build any graph
            self._builder = DistributedReplicatedBuilder(gpus, server)
            self.is_chief = self._builder.is_chief
        else:
            self.is_chief = False
        logger.info("Distributed training on cluster:\n" + str(server.server_def.cluster))

    def _setup_input(self, inputs_desc, input):
        if self.job_name == 'ps':
            # ps shouldn't setup input either
            logger.info("Running ps {}".format(self.server.server_def.task_index))
            logger.info("Kill me with 'kill {}'".format(os.getpid()))
            self.server.join()  # this function will never return tensorflow#4713
            raise RuntimeError("This is a bug in tensorpack. Server.join() for ps should never return!")

        with override_to_local_variable():
            get_global_step_var()  # gs should be local
            # input source may create variable (queue size summary)
            # TODO This is not good because we don't know from here
            # whether something should be global or local. We now assume
            # they should be local.
            assert not input.setup_done()
            return input.setup(inputs_desc)

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        self.train_op, initial_sync_op, model_sync_op = self._builder.build(
            self._make_get_grad_fn(input, get_cost_fn, get_opt_fn), get_opt_fn)

        callbacks = []
        # initial local_vars syncing
        cb = RunOp(lambda: initial_sync_op,
                   run_before=True, run_as_trigger=False, verbose=True)
        cb.chief_only = False
        callbacks.append(cb)

        # model_variables syncing
        if model_sync_op:
            cb = RunOp(lambda: model_sync_op,
                       run_before=False, run_as_trigger=True, verbose=True)
            logger.warn("For efficiency, local MODEL_VARIABLES are only synced to PS once "
                        "every epoch. Be careful if you save the model more frequently than this.")
            callbacks.append(cb)
        return callbacks

    @HIDE_DOC
    def initialize(self, session_creator, session_init):
        if not isinstance(session_creator, NewSessionCreator) or \
                session_creator.user_provided_config:
            raise ValueError(
                "You are not allowed to set session_creator or session_config for distributed training! "
                "To use a custom session config, pass it to tf.train.Server.")
        super(DistributedTrainerReplicated, self).initialize(
            get_distributed_session_creator(), session_init)

    @property
    def _main_tower_vs_name(self):
        return "tower0"


class HorovodTrainer(SingleCostTrainer):
    """
    Horovod trainer, support multi-GPU and distributed training.

    To use for multi-GPU training:

        CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 --output-filename mylog python train.py

    To use for distributed training:

        /path/to/mpirun -np 8 -H server1:4,server2:4  \
            -bind-to none -map-by slot \
            --output-filename mylog  -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
            python train.py

    Note:
        1. If using all GPUs, you can always skip the `CUDA_VISIBLE_DEVICES` option.

        2. About performance, horovod is expected to be slightly
           slower than native tensorflow on multi-GPU training, but faster in distributed training.

        3. Due to the use of MPI, training is less informative (no progress bar).
           It's recommended to use other multi-GPU trainers for single-node
           experiments, and scale to multi nodes by horovod.
    """
    def __init__(self):
        hvd.init()
        self.is_chief = hvd.rank() == 0
        self._local_rank = hvd.local_rank()
        logger.info("Horovod local rank={}".format(self._local_rank))
        super(HorovodTrainer, self).__init__()

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        with TowerContext('', is_training=True):
            grads = self._make_get_grad_fn(input, get_cost_fn, get_opt_fn)()
            opt = get_opt_fn()
            opt = hvd.DistributedOptimizer(opt)
            self.train_op = opt.apply_gradients(grads, name='min_op')
        with tf.name_scope('horovod_broadcast'):
            op = hvd.broadcast_global_variables(0)
        cb = RunOp(
            op, run_before=True,
            run_as_trigger=False, verbose=True)
        cb.chief_only = False
        return [cb]

    @HIDE_DOC
    def initialize(self, session_creator, session_init):
        if not isinstance(session_creator, NewSessionCreator):
            raise ValueError(
                "session_creator has to be `NewSessionCreator` for horovod training! ")
        session_creator.config.gpu_options.visible_device_list = str(self._local_rank)
        super(HorovodTrainer, self).initialize(
            session_creator, session_init)


from ..utils.develop import create_dummy_class   # noqa
try:
    import horovod.tensorflow as hvd
except ImportError:
    HorovodTrainer = create_dummy_class('HovorodTrainer', 'horovod')    # noqa
except Exception:      # could be other than ImportError, e.g. NCCL not found
    print("Horovod is installed but cannot be imported.")
    HorovodTrainer = create_dummy_class('HovorodTrainer', 'horovod')    # noqa
