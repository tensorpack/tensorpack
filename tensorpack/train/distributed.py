#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distributed.py

import tensorflow as tf
import re
import os
from six.moves import range

from ..utils import logger
from ..callbacks import RunOp
from ..tfutils.sesscreate import NewSessionCreator
from ..tfutils.common import get_global_step_var, get_op_tensor_name

from .multigpu import MultiGPUTrainerBase
from .utility import override_to_local_variable


__all__ = ['DistributedTrainerReplicated']


class DistributedTrainerReplicated(MultiGPUTrainerBase):
    """
    Distributed replicated training.
    Each worker process builds the same model on one or more GPUs.
    Gradients across GPUs are averaged within the worker,
    and get synchronously applied to the global copy of variables located on PS.
    Then each worker copy the latest variables from PS back to local.

    See https://www.tensorflow.org/performance/benchmarks for details.

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
        server_def = server.server_def
        self.cluster = tf.train.ClusterSpec(server_def.cluster)
        self.job_name = server_def.job_name
        self.task_index = server_def.task_index
        assert self.job_name in ['ps', 'worker'], self.job_name
        assert tf.test.is_gpu_available
        logger.info("Distributed training on cluster:\n" + str(server_def.cluster))
        logger.info("My role in the cluster: job={}, task={}".format(self.job_name, self.task_index))

        self._input_source = config.data
        self.is_chief = (self.task_index == 0 and self.job_name == 'worker')

        worker_prefix = '/job:worker/task:%s' % self.task_index
        self.param_server_device = tf.train.replica_device_setter(
            worker_device=worker_prefix + '/cpu:0', cluster=self.cluster)
        self.num_ps = self.cluster.num_tasks('ps')
        self.num_worker = self.cluster.num_tasks('worker')

        self.nr_gpu = config.nr_tower
        self.cpu_device = '%s/cpu:0' % worker_prefix
        self.raw_devices = ['%s/%s:%i' % (worker_prefix, 'gpu', i) for i in range(self.nr_gpu)]

        # Device for queues for managing synchronization between servers
        self.sync_queue_devices = ['/job:ps/task:%s/cpu:0' % i for i in range(self.num_ps)]
        self.sync_queue_counter = 0

        super(DistributedTrainerReplicated, self).__init__(config)

    @staticmethod
    def _average_grads(tower_grads, devices):
        """
        Average grads from towers.
        The device where the average happens is chosen with round-robin.

        Args:
            tower_grads: Ngpu x Nvar x 2

        Returns:
            Nvar x 2
        """
        nr_device = len(devices)
        if nr_device == 1:
            return tower_grads[0]
        new_tower_grads = []
        with tf.name_scope('AvgGrad'):
            for i, grad_and_vars in enumerate(zip(*tower_grads)):
                # Ngpu * 2
                with tf.device(devices[i % nr_device]):
                    v = grad_and_vars[0][1]
                    # average gradient
                    all_grads = [g for (g, _) in grad_and_vars]
                    grad = tf.multiply(
                        tf.add_n(all_grads), 1.0 / nr_device)
                    new_tower_grads.append((grad, v))
        return new_tower_grads

    @staticmethod
    def _apply_shadow_vars(avg_grads):
        """
        Create shadow variables on PS, and replace variables in avg_grads
        by these shadow variables.

        Args:
            avg_grads: list of (grad, var) tuples
        """
        ps_var_grads = []
        for grad, var in avg_grads:
            assert var.name.startswith('tower'), var.name
            my_name = '/'.join(var.name.split('/')[1:])
            my_name = get_op_tensor_name(my_name)[0]
            new_v = tf.get_variable(my_name, dtype=var.dtype.base_dtype,
                                    initializer=var.initial_value,
                                    trainable=True)
            # (g, v) to be applied, where v is global (ps vars)
            ps_var_grads.append((grad, new_v))
        return ps_var_grads

    @staticmethod
    def _shadow_model_variables(shadow_vars):
        """
        Create shadow vars for model_variables as well, and add to the list of ``shadow_vars``.

        Returns:
            list of (shadow_model_var, local_model_var) used for syncing.
        """
        curr_shadow_vars = set([v.name for v in shadow_vars])
        model_vars = tf.model_variables()
        shadow_model_vars = []
        for v in model_vars:
            assert v.name.startswith('tower'), "Found some MODEL_VARIABLES created outside of the model!"
            stripped_name = get_op_tensor_name(re.sub('tower[0-9]+/', '', v.name))[0]
            if stripped_name in curr_shadow_vars:
                continue
            new_v = tf.get_variable(stripped_name, dtype=v.dtype.base_dtype,
                                    initializer=v.initial_value,
                                    trainable=False)

            curr_shadow_vars.add(stripped_name)  # avoid duplicated shadow_model_vars
            shadow_vars.append(new_v)
            shadow_model_vars.append((new_v, v))  # only need to sync model_var from one tower
        return shadow_model_vars

    def _apply_gradients_and_copy(self, raw_grad_list, ps_var_grads):
        """
        Apply averaged gradients to ps vars, and then copy the updated
        variables back to each tower.

        Args:
            raw_grad_list: Ngpu x Nvar x 2 gradient list from all towers
            ps_var_grads: Nvar x 2 (grad, ps_var)

        Returns:
            list of copy ops
        """
        # TODO do this for variables together?
        opt = self.model.get_optimizer()
        var_update_ops = []
        for vid, (g, v) in enumerate(ps_var_grads):
            # TODO do we put momentum variables into local or global?
            apply_gradient_op = opt.apply_gradients([(g, v)])
            barrier = self._add_sync_queues_and_barrier(
                'param_update_barrier_{}'.format(vid), [apply_gradient_op])
            with tf.control_dependencies([barrier]), \
                    tf.device(self.cpu_device):
                updated_value = v.read_value()
                for towerid in range(self.nr_gpu):
                    var_update_ops.append(
                        raw_grad_list[towerid][vid][1].assign(updated_value))
        return var_update_ops

    def _setup(self):
        if self.job_name == 'ps':
            logger.info("Running ps {}".format(self.task_index))
            logger.info("Kill me with 'kill {}'".format(os.getpid()))
            self.server.join()  # this will never return tensorflow#4713
            return
        with tf.device(self.param_server_device):
            gs = get_global_step_var()
            assert gs.device, gs.device
        # do this before inputsource.setup because input_source my need global step

        with override_to_local_variable():
            # input source may create variable (queue size summary)
            # TODO This is not good because we don't know from here
            # whether something should be global or local. We now assume
            # they should be local.
            cbs = self._input_source.setup(self.model.get_inputs_desc())
        self.config.callbacks.extend(cbs)

        # Build the optimizer first, before entering any tower.
        # This makes sure that learning_rate is a global variable (what we expect)
        self.model.get_optimizer()

        # Ngpu * Nvar * 2
        grad_list = MultiGPUTrainerBase.build_on_multi_tower(
            self.config.tower,
            lambda: MultiGPUTrainerBase._build_graph_get_grads(
                self.model, self._input_source),
            devices=self.raw_devices,
            use_vs=[True] * self.config.nr_tower)  # open vs at each tower
        MultiGPUTrainerBase._check_grad_list(grad_list)

        avg_grads = DistributedTrainerReplicated._average_grads(grad_list, self.raw_devices)
        with tf.device(self.param_server_device):
            ps_var_grads = DistributedTrainerReplicated._apply_shadow_vars(avg_grads)
            var_update_ops = self._apply_gradients_and_copy(grad_list, ps_var_grads)
            self._shadow_vars = [v for (_, v) in ps_var_grads]
            self._shadow_model_vars = DistributedTrainerReplicated._shadow_model_variables(self._shadow_vars)

        # TODO add options to synchronize less
        main_fetch = tf.group(*var_update_ops, name='main_fetches')
        self.train_op = self._add_sync_queues_and_barrier(
            'post_copy_barrier', [main_fetch])

        # initial local_vars syncing
        cb = RunOp(self._get_initial_sync_op,
                   run_before=True, run_as_trigger=False, verbose=True)
        cb.chief_only = False
        self.register_callback(cb)

        # model_variables syncing
        if len(self._shadow_model_vars) and self.is_chief:
            cb = RunOp(self._get_sync_model_vars_op,
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

    def _add_sync_queues_and_barrier(self, name, dependencies):
        """Adds ops to enqueue on all worker queues.

        Args:
            name: prefixed for the shared_name of ops.
            dependencies: control dependency from ops.

        Returns:
            an op that should be used as control dependency before starting next step.
        """
        self.sync_queue_counter += 1
        with tf.device(self.sync_queue_devices[self.sync_queue_counter % len(self.sync_queue_devices)]):
            sync_queues = [
                tf.FIFOQueue(self.num_worker, [tf.bool], shapes=[[]],
                             shared_name='%s%s' % (name, i))
                for i in range(self.num_worker)]
            queue_ops = []
            # For each other worker, add an entry in a queue, signaling that it can finish this step.
            token = tf.constant(False)
            with tf.control_dependencies(dependencies):
                for i, q in enumerate(sync_queues):
                    if i != self.task_index:
                        queue_ops.append(q.enqueue(token))

            # Drain tokens off queue for this worker, one for each other worker.
            queue_ops.append(
                sync_queues[self.task_index].dequeue_many(len(sync_queues) - 1))

            return tf.group(*queue_ops, name=name)

    def _get_initial_sync_op(self):
        """
        Get the op to copy-initialized all local variables from PS.
        """
        def strip_port(s):
            if s.endswith(':0'):
                return s[:-2]
            return s
        local_vars = tf.local_variables()
        local_var_by_name = dict([(strip_port(v.name), v) for v in local_vars])
        ops = []
        nr_shadow_vars = len(self._shadow_vars)
        for v in self._shadow_vars:
            vname = strip_port(v.name)
            for i in range(self.nr_gpu):
                name = 'tower%s/%s' % (i, vname)
                assert name in local_var_by_name, \
                    "Shadow variable {} doesn't match a corresponding local variable!".format(v.name)
                copy_to = local_var_by_name[name]
                # logger.info("{} -> {}".format(v.name, copy_to.name))
                ops.append(copy_to.assign(v.read_value()))
        return tf.group(*ops, name='sync_{}_variables_from_ps'.format(nr_shadow_vars))

    def _get_sync_model_vars_op(self):
        """
        Get the op to sync local model_variables to PS.
        """
        ops = []
        for (shadow_v, local_v) in self._shadow_model_vars:
            ops.append(shadow_v.assign(local_v.read_value()))
        assert len(ops)
        return tf.group(*ops, name='sync_{}_model_variables_to_ps'.format(len(ops)))

    @property
    def vs_name_for_predictor(self):
        return "tower0"
