#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distributed.py

import tensorflow as tf
import os

from ..utils import logger
from ..callbacks import RunOp
from ..tfutils.sesscreate import NewSessionCreator
from ..tfutils.common import get_global_step_var

from ..graph_builder.distributed import DistributedReplicatedBuilder
from ..graph_builder.utils import override_to_local_variable
from .base import Trainer


__all__ = ['DistributedTrainerReplicated']


class DistributedTrainerReplicated(Trainer):
    """
    Build the graph with :class:`DistributedReplicatedBuilder` and train it.

    Note:
        Gradients are not averaged across workers, but applied to PS variables
        directly (either with or without locking depending on the optimizer).

    Example:

        .. code-block:: python

            hosts = ['host1.com', 'host2.com']
            cluster_spec = tf.train.ClusterSpec({
                'ps': [h + ':2222' for h in hosts],
                'worker': [h + ':2223' for h in hosts]
            })
            server = tf.train.Server(
                cluster_spec, job_name=args.job, task_index=args.task,
                config=get_default_sess_config())
            DistributedTrainerReplicated(config, server).train()

        .. code-block:: none

            # start your jobs:
            (host1)$ train.py --job worker --task 0
            (host1)$ train.py --job ps --task 0
            (host2)$ train.py --job worker --task 1
            (host2)$ train.py --job ps --task 1
    """
    def __init__(self, config, server):
        """
        Args:
            config(TrainConfig): Must contain 'model' and 'data'.
            server(tf.train.Server): the server object with ps and workers
        """
        assert config.data is not None and config.model is not None

        self.server = server
        self.job_name = server.server_def.job_name
        assert self.job_name in ['ps', 'worker'], self.job_name

        if self.job_name == 'worker':
            # ps doesn't build any graph
            self._builder = DistributedReplicatedBuilder(config.tower, server)
            self.is_chief = self._builder.is_chief
        else:
            self.is_chief = False
        logger.info("Distributed training on cluster:\n" + str(server.server_def.cluster))

        self._input_source = config.data
        self.nr_gpu = config.nr_tower

        super(DistributedTrainerReplicated, self).__init__(config)

    def _setup(self):
        if self.job_name == 'ps':
            logger.info("Running ps {}".format(self._builder.task_index))
            logger.info("Kill me with 'kill {}'".format(os.getpid()))
            self.server.join()  # this will never return tensorflow#4713
            return

        # always do this before inputsource.setup because input_source my need global step
        # TODO Can we just do this in get_global_step_var
        with tf.device(self._builder.param_server_device):
            gs = get_global_step_var()
            assert gs.device, gs.device

        with override_to_local_variable():
            # input source may create variable (queue size summary)
            # TODO This is not good because we don't know from here
            # whether something should be global or local. We now assume
            # they should be local.
            cbs = self._input_source.setup(self.model.get_inputs_desc())
        self.config.callbacks.extend(cbs)

        self.train_op, initial_sync_op, model_sync_op = self._builder.build(
            self._input_source, self.model.build_graph_get_cost, self.model.get_optimizer)

        # initial local_vars syncing
        cb = RunOp(lambda: initial_sync_op,
                   run_before=True, run_as_trigger=False, verbose=True)
        cb.chief_only = False
        self.register_callback(cb)

        # model_variables syncing
        if model_sync_op:
            cb = RunOp(lambda: model_sync_op,
                       run_before=False, run_as_trigger=True, verbose=True)
            logger.warn("For efficiency, local MODEL_VARIABLES are only synced to PS once "
                        "every epoch. Be careful if you save the model more frequently than this.")
            self.register_callback(cb)

        self._set_session_creator()

    def _set_session_creator(self):
        old_sess_creator = self.config.session_creator
        if not isinstance(old_sess_creator, NewSessionCreator) \
                or self.config.session_config is not None:
            raise ValueError(
                "Cannot set session_creator or session_config for distributed training! "
                "To use a custom session config, pass it with tf.train.Server.")

        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()
        ready_op = tf.report_uninitialized_variables()
        sm = tf.train.SessionManager(
            local_init_op=local_init_op,
            ready_op=ready_op, graph=tf.get_default_graph())

        # to debug wrong variable collection
        # print("GLOBAL:")
        # print(tf.global_variables())
        # print("LOCAL:")
        # print(tf.local_variables())
        def _create_session():
            if self.is_chief:
                return sm.prepare_session(master=self.server.target, init_op=init_op)
            else:
                return sm.wait_for_session(master=self.server.target)

        class _Creator(tf.train.SessionCreator):
            def create_session(self):
                return _create_session()

        self.config.session_creator = _Creator()

    @property
    def vs_name_for_predictor(self):
        return "tower0"
