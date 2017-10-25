#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distributed.py

import os

from ..utils import logger
from ..callbacks import RunOp
from ..tfutils.sesscreate import NewSessionCreator
from ..tfutils import get_global_step_var
from ..tfutils.distributed import get_distributed_session_creator

from ..graph_builder.distributed import DistributedReplicatedBuilder
from ..graph_builder.utils import override_to_local_variable
from .base import Trainer


__all__ = ['DistributedTrainerReplicated']


class DistributedTrainerReplicated(Trainer):

    __doc__ = DistributedReplicatedBuilder.__doc__

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

        super(DistributedTrainerReplicated, self).__init__(config)

    def _setup(self):
        if self.job_name == 'ps':
            logger.info("Running ps {}".format(self.server.server_def.task_index))
            logger.info("Kill me with 'kill {}'".format(os.getpid()))
            self.server.join()  # this will never return tensorflow#4713
            return

        with override_to_local_variable():
            get_global_step_var()  # gs should be local

            # input source may create variable (queue size summary)
            # TODO This is not good because we don't know from here
            # whether something should be global or local. We now assume
            # they should be local.
            cbs = self._input_source.setup(self.model.get_inputs_desc())
        self._config.callbacks.extend(cbs)

        self.train_op, initial_sync_op, model_sync_op = self._builder.build(
            lambda: self.model._build_graph_get_grads(
                *self._input_source.get_input_tensors()),
            self.model.get_optimizer)

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
        old_sess_creator = self._config.session_creator
        if not isinstance(old_sess_creator, NewSessionCreator) \
                or old_sess_creator.user_provided_config:
            raise ValueError(
                "Cannot set session_creator or session_config for distributed training! "
                "To use a custom session config, pass it to tf.train.Server.")

        self._config.session_creator = get_distributed_session_creator(self.server)

    @property
    def _main_tower_vs_name(self):
        return "tower0"
