#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distributed.py

import tensorflow as tf
import re
from six.moves import zip, range

from ..utils.argtools import memoized
from ..tfutils.common import get_op_tensor_name, get_global_step_var

from .training import DataParallelBuilder
from .utils import override_to_local_variable

__all__ = ['DistributedReplicatedBuilder']


class DistributedReplicatedBuilder(DataParallelBuilder):
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

            # Create the server object like this:
            hosts = ['host1.com', 'host2.com']
            cluster_spec = tf.train.ClusterSpec({
                'ps': [h + ':2222' for h in hosts],
                'worker': [h + ':2223' for h in hosts]
            })
            server = tf.train.Server(
                cluster_spec, job_name=args.job, task_index=args.task,
                config=get_default_sess_config())
            # initialize trainer with this server object

        .. code-block:: none

            # Start training like this:
            (host1)$ train.py --job worker --task 0
            (host1)$ train.py --job ps --task 0
            (host2)$ train.py --job worker --task 1
            (host2)$ train.py --job ps --task 1
    """

    def __init__(self, towers, server):
        """
        Args:
            towers (list[int]): list of GPU ids.
            server (tf.train.Server): the server with ps and workers.
                The job_name must be 'worker' because 'ps' job doesn't need to
                build any graph.
        """
        super(DistributedReplicatedBuilder, self).__init__(towers)
        self.server = server
        server_def = server.server_def
        self.cluster = tf.train.ClusterSpec(server_def.cluster)
        self.task_index = server_def.task_index

        self.is_chief = (self.task_index == 0)

        worker_prefix = '/job:worker/task:%s' % self.task_index
        self.param_server_device = tf.train.replica_device_setter(
            worker_device=worker_prefix + '/cpu:0', cluster=self.cluster)
        self.num_ps = self.cluster.num_tasks('ps')
        self.num_worker = self.cluster.num_tasks('worker')

        self.nr_gpu = len(self.towers)
        self.cpu_device = '%s/cpu:0' % worker_prefix
        self.raw_devices = ['%s/%s:%i' % (worker_prefix, 'gpu', i) for i in range(self.nr_gpu)]

        # Device for queues for managing synchronization between servers
        self.sync_queue_devices = ['/job:ps/task:%s/cpu:0' % i for i in range(self.num_ps)]
        self.sync_queue_counter = 0

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
                v = grad_and_vars[0][1]  # Ngpu * 2
                all_grads = [g for (g, _) in grad_and_vars]
                with tf.device(devices[i % nr_device]):
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

    def build(self, get_grad_fn, get_opt_fn):
        """
        Args:
            get_grad_fn (-> [(grad, var)]):
            get_opt_fn (-> tf.train.Optimizer): callable which returns an optimizer

        Returns:
            (tf.Operation, tf.Operation, tf.Operation):

            1. the training op.

            2. the op which sync all the local variables from PS.
            This op sholud be run before training.

            3. the op which sync all the local `MODEL_VARIABLES` from PS.
            You can choose how often to run it by yourself.
        """
        with override_to_local_variable():
            get_global_step_var()

        get_opt_fn = memoized(get_opt_fn)
        # Build the optimizer first, before entering any tower.
        # This makes sure that learning_rate is a global variable (what we expect)
        get_opt_fn()    # TODO get_opt_fn called before main graph was built

        # Ngpu * Nvar * 2
        grad_list = DataParallelBuilder.build_on_towers(
            self.towers, get_grad_fn,
            devices=self.raw_devices,
            use_vs=[True] * len(self.towers))  # open vs at each tower
        DataParallelBuilder._check_grad_list(grad_list)

        avg_grads = DistributedReplicatedBuilder._average_grads(grad_list, self.raw_devices)
        with tf.device(self.param_server_device):
            ps_var_grads = DistributedReplicatedBuilder._apply_shadow_vars(avg_grads)
            var_update_ops = self._apply_gradients_and_copy(
                get_opt_fn(), grad_list, ps_var_grads)
            self._shadow_vars = [v for (__, v) in ps_var_grads]
            self._shadow_model_vars = DistributedReplicatedBuilder._shadow_model_variables(self._shadow_vars)

        # TODO add options to synchronize less
        main_fetch = tf.group(*var_update_ops, name='main_fetches')
        train_op = self._add_sync_queues_and_barrier(
            'post_copy_barrier', [main_fetch])

        # initial local_vars syncing
        initial_sync_op = self._get_initial_sync_op()
        if len(self._shadow_model_vars) and self.is_chief:
            model_sync_op = self._get_sync_model_vars_op()
        else:
            model_sync_op = None
        return train_op, initial_sync_op, model_sync_op

    def _apply_gradients_and_copy(self, opt, raw_grad_list, ps_var_grads):
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
