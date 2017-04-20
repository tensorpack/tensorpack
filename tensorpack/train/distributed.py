#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py

import tensorflow as tf
from .input_data import QueueInput
import weakref

from tensorflow.python.training.monitored_session \
    import _HookedSession as HookedSession

from .base import Trainer
from .multigpu import SyncMultiGPUTrainer
from ..utils import logger

from ..callbacks.monitor import Monitors
from ..tfutils.model_utils import describe_model
from ..callbacks import Callbacks
from ..tfutils.common import get_global_step_var


__all__ = ['DistributedTrainer']


class DistributedTrainer(Trainer):
    """
    A multi-device multi-GPU trainer.

    """

    def __init__(self, config, task_index, job_name, input_queue=None):
        """Initialize a distributed version for training a network across different machines

        Args:
            config (TrainConfig): same as in :class:`QueueInputTrainer`.
            task_index (int): identifier for current task
            job_name (str): parameter server "ps" or "worker"
            input_queue (None, optional): same as in :class:`QueueInputTrainer`.

        Example:
        The config should contain the cluster_spec as described in the TF documentation
            config = {
                cluster_spec={
                    'ps': ['machineA:2222'],
                    'worker': ['machineA:2223','machineB:2224']
                }
            }
        """
        if config.dataflow is not None:
            self._input_method = QueueInput(config.dataflow, input_queue)
        else:
            self._input_method = config.data

        super(DistributedTrainer, self).__init__(config)

        # for convenience
        cluster_config = config.cluster_spec
        assert cluster_config, "distributed version requires a cluster_spec entry"

        # some sanity checks
        errmsg = "they are only two kinds of jobs ('ps', 'worker') but '{}' was given".format(job_name)
        assert job_name in ["ps", "worker"], errmsg

        if job_name == "ps":
            max_num = len(cluster_config['ps'])
            err_msg = "task_index should be less than %i but is $%i" % (max_num, task_index)
            assert task_index < max_num, err_msg
        else:
            max_num = len(cluster_config['worker'])
            err_msg = "task_index should be less than %i but is $%i" % (max_num, task_index)
            assert task_index < max_num, err_msg

        if len(config.tower) > 1:
            assert tf.test.is_gpu_available()

        # specify cluster layout
        self.cluster_spec = tf.train.ClusterSpec(cluster_config)

        # specify current entity
        self.server = tf.train.Server(self.cluster_spec, job_name=job_name, task_index=task_index)
        self.task_index = task_index
        self.job_name = job_name

    def _setup(self):
        pass

    def run_step(self):
        self.hooked_sess.run(self.train_op)

    def setup(self):

        self.monitors = Monitors(self.monitors)
        self.register_callback(self.monitors)

        describe_model()

        # some final operations that might modify the graph
        logger.info("Setup callbacks graph ...")
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

        is_chief = (self.task_index == 0)
        num_workers = len(self.config.cluster_spec['worker'])

        if self.job_name == "ps":
            self.server.join()
        else:
            # build model etc
            logger.info("Building worker {}...".format(self.config.cluster_spec['worker'][self.task_index]))
            with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % self.task_index,
                                                          cluster=self.cluster_spec)):
                device_trainer = SyncMultiGPUTrainer(self.config)

                opt = device_trainer.config.optimizer
                # we omit the "setup()"-part
                device_trainer._setup()

                init_token_op = opt.get_init_tokens_op()
                chief_queue_runner = opt.get_chief_queue_runner()

                global_step = get_global_step_var()

                # init session
                init_op = tf.global_variables_initializer()

                logger.info("Graph variables initialized.")
                self.train_op = opt.apply_gradients(device_trainer.grads, name='min_op', global_step=global_step)

            config_proto = tf.ConfigProto(allow_soft_placement=True)
            sv = tf.train.Supervisor(is_chief=is_chief, init_op=init_op)
            self.sess = sv.prepare_or_wait_for_session(self.server.target, config=config_proto)

            if is_chief:
                sv.start_queue_runners(self.sess, [chief_queue_runner])
                self.sess.run(init_token_op)

        hooks = self._callbacks.get_hooks()
        self.hooked_sess = HookedSession(self.sess, hooks)

        # create session
        logger.info("Finalize the graph, create the session ...")
        # self.sess.graph.finalize()
