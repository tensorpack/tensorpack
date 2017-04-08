#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py

import tensorflow as tf
from .input_data import QueueInput

from .base import Trainer
from .multigpu import SyncMultiGPUTrainer
from ..utils import logger


class DistributedTrainer(Trainer):
    """
    A multi-device multi-GPU trainer.

    """

    def __init__(self, config, task_index, job_name, input_queue=None):
        """Initialize a distributed version for training a network (sync-version)

        Args:
            config (TYPE): Description
            task_index (TYPE): Description
            job_name (TYPE): Description
            input_queue (None, optional): Description

        Example:
        An example of config is
            config = {
                'cluster':{
                    'ps': ['machineA:2222'],
                    'worker': [
                        {'host': 'machineA:2223', 'gpu': [0]},
                        {'host': 'machineB:2224', 'gpu': [1]}
                    ]
                }
            }
        """
        if config.dataflow is not None:
            self._input_method = QueueInput(config.dataflow, input_queue)
        else:
            self._input_method = config.data

        super(Trainer, self).__init__(config)

        # for convenience
        cluster_config = config['cluster']

        # some sanity checks
        assert len(config.tower) >= 1, "Distributed trainer must be used with at least one tower."
        assert len(cluster_config['ps']) >= 1, "Distributed trainer requires at least one parameter server."
        assert len(cluster_config['worker']) >= 1, "Distributed trainer requires at least one worker."
        errmsg = "they are only two kinds of jobs ('ps', 'worker') but {} was given".format(job_name)
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
        ps_hosts = cluster_config['ps']
        worker_hosts = [k['host'] for k in cluster_config['worker']]
        self.cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

        # specify current entity
        self.server = tf.train.Server(self.cluster_spec, job_name=job_name, task_index=task_index)
        self.task_index = task_index
        self.job_name = job_name
        self.num_workers = len(worker_hosts)

    def _setup(self):
        super(DistributedTrainer, self)._setup()

        cluster_config = self.config['cluster']
        is_chief = (self.task_index == 0)

        if self.job_name == "ps":
            self.server.join()
        else:
            # build model etc
            logger.info("Building worker {}...".format(cluster_config['worker'][self.task_index]))
            with tf.Graph().as_default():
                with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % self.task_index,
                                                              cluster=self.cluster_spec)):
                    device_trainer = SyncMultiGPUTrainer(self.config)
                    # TODO: is this place the best spot for this code?
                    opt = device_trainer.config.optimizer
                    opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=self.num_workers,
                                                         total_num_replicas=self.num_workers)
                    device_trainer.config.optimizer = opt
                    device_trainer._setup()

                    init_token_op = opt.get_init_tokens_op()
                    chief_queue_runner = opt.get_chief_queue_runner()

                    # this should be actually part of sessinit.py or sesscreate.py
                    global_step = tf.train.get_global_step()  # really?
                    init = tf.global_variables_initializer()
                    sv = tf.train.Supervisor(is_chief=is_chief,
                                             init_op=init,
                                             global_step=global_step)
                    # Create a session for running Ops on the Graph.
                    config_proto = tf.ConfigProto(allow_soft_placement=True)
                    sess = sv.prepare_or_wait_for_session(self.server.target, config=config_proto)

                    # TODO: create a new sess-init class for distributed setting
                    self.train_op = opt.apply_gradients(device_trainer.grads, name='min_op')

                if is_chief:
                    sv.start_queue_runners(sess, [chief_queue_runner])
                    sess.run(init_token_op)
