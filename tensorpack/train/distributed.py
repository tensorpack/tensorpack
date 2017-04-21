#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distributed.py

import tensorflow as tf
from six.moves import range
import weakref
from tensorflow.python.training.monitored_session \
    import _HookedSession as HookedSession

from ..utils import logger
from .input_source import StagingInputWrapper, FeedfreeInput
from .feedfree import SingleCostFeedfreeTrainer
from .multigpu import MultiGPUTrainerBase
from ..tfutils.model_utils import describe_model
from ..callbacks import Callbacks, ProgressBar
from ..tfutils.sesscreate import ReuseSessionCreator
from ..tfutils.common import get_default_sess_config, get_global_step_var
from ..callbacks.monitor import Monitors

__all__ = ['DistributedReplicatedTrainer']

PS_SHADOW_VAR_PREFIX = 'ps_var'


# To be used with custom_getter on tf.get_variable. Ensures the created variable
# is in LOCAL_VARIABLES and not GLOBAL_VARIBLES collection.
class OverrideToLocalVariableIfNotPsVar(object):

    # args and kwargs come from the custom_getter interface for Tensorflow
    # variables, and matches tf.get_variable's signature, with the addition of
    # 'getter' at the beginning.
    def __call__(self, getter, name, *args, **kwargs):
        if name.startswith(PS_SHADOW_VAR_PREFIX):
            return getter(*args, **kwargs)
        logger.info("CustomGetter-{}".format(name))

        if 'collections' in kwargs:
            collections = kwargs['collections']
        if not collections:
            collections = set([tf.GraphKeys.GLOBAL_VARIABLES])
        else:
            collections = set(collections.copy())
        collections.remove(tf.GraphKeys.GLOBAL_VARIABLES)
        collections.add(tf.GraphKeys.LOCAL_VARIABLES)
        kwargs['collections'] = list(collections)
        return getter(name, *args, **kwargs)


class DistributedReplicatedTrainer(SingleCostFeedfreeTrainer):
    def __init__(self, config, job_name, task_index, cluster):
        assert job_name in ['ps', 'worker'], job_name
        self.config = config
        self.job_name = job_name
        self.task_index = task_index
        self.cluster = cluster
        self._input_source = config.data
        super(DistributedReplicatedTrainer, self).__init__(config)

        worker_prefix = '/job:worker/task:%s' % self.task_index
        self.param_server_device = tf.train.replica_device_setter(
            worker_device=worker_prefix + '/cpu:0', cluster=self.cluster)
        # This device on which the queues for managing synchronization between
        # servers should be stored.
        num_ps = self.cluster.num_tasks('ps')

        self.cpu_device = '%s/cpu:0' % worker_prefix
        self.nr_gpu = config.nr_tower
        self.raw_devices = ['%s/%s:%i' % (worker_prefix, 'gpu', i) for i in range(self.nr_gpu)]
        self.sync_queue_devices = ['/job:ps/task:%s/cpu:0' % i for i in range(num_ps)]
        self.sync_queue_counter = 0

        if self.nr_gpu > 1:
            assert tf.test.is_gpu_available()

            # seem to only improve on >1 GPUs
            if not isinstance(self._input_source, StagingInputWrapper):
                self._input_source = StagingInputWrapper(self._input_source, self.raw_devices)

    def _setup(self):
        conf = get_default_sess_config()
        self.server = tf.train.Server(
            self.cluster, job_name=self.job_name,
            task_index=self.task_index,
            config=conf # TODO sessconfig
        )

        if self.job_name == 'ps':
            logger.info("Running ps {}".format(self.task_index))
            self.server.join()
            return
        with tf.variable_scope(
                tf.get_variable_scope(),
                custom_getter=OverrideToLocalVariableIfNotPsVar()):
            # Ngpu * Nvar * 2
            grad_list = MultiGPUTrainerBase.build_on_multi_tower(
                self.config.tower,
                lambda: self._get_cost_and_grad()[1],
                devices=self.raw_devices,
                var_strategy='replicated')

        # (g, v) to be applied, where v is global (ps vars)
        new_tower_grads = []
        for i, grad_and_vars in enumerate(zip(*grad_list)):
            # Ngpu * 2
            with tf.device(self.raw_devices[i % self.nr_gpu]):
                v = grad_and_vars[0][1]
                if self.nr_gpu > 1:
                    # average gradient
                    all_grads = [g for (g, _) in grad_and_vars]
                    if not MultiGPUTrainerBase.check_none_grads(v.op.name, all_grads):
                        continue
                    grad = tf.multiply(
                        tf.add_n(all_grads), 1.0 / self.nr_gpu)
                else:
                    grad = grad_and_vars[0][0]

            with tf.device(self.param_server_device):
                my_name = PS_SHADOW_VAR_PREFIX + '/' + v.name
                if my_name.endswith(':0'):
                    my_name = my_name[:-2]
                new_v = tf.get_variable(my_name, dtype=v.dtype.base_dtype,
                                        initializer=v.initial_value,
                                        trainable=True)
                new_tower_grads.append((grad, new_v))

        # apply gradients TODO do this for each variable separately?
        opt = self.model.get_optimizer()
        apply_gradient_op = opt.apply_gradients(new_tower_grads)
        barrier = self.add_sync_queues_and_barrier('replicate_variable', [apply_gradient_op])
        var_update_ops = []
        with tf.control_dependencies([barrier]), \
                tf.device(self.cpu_device):
            for idx, (grad, v) in enumerate(new_tower_grads):
                updated_value = v.read_value()
                for towerid in range(self.nr_gpu):
                    logger.info("Step update {} -> {}".format(v.name, grad_list[towerid][idx][1].name))
                    var_update_ops.append(
                        grad_list[towerid][idx][1].assign(updated_value))
        self.main_fetch = tf.group(*var_update_ops, name='main_fetches')
        self.train_op = self.add_sync_queues_and_barrier('sync_queues_step_end', [self.main_fetch])
        self.post_init_op = self.get_post_init_ops()

    def setup(self):
        with tf.device(self.param_server_device):
            gs = get_global_step_var()
        self.is_chief = (self.task_index == 0 and self.job_name == 'worker')
        assert isinstance(self._input_source, FeedfreeInput), type(self._input_source)
        self._input_source.setup_training(self)

        self._setup()

        self.monitors = Monitors(self.monitors)
        self.register_callback(self.monitors)
        describe_model()
        # some final operations that might modify the graph
        logger.info("Setup callbacks graph ...")

        #if not self.is_chief:
            #self._callbacks = [ProgressBar()]
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

        #local_init_op = tf.local_variables_initializer()
        global_init_op = tf.global_variables_initializer()

        logger.info("Finalize the graph, create the session ...")

        self.sv = tf.train.Supervisor(
            is_chief=self.is_chief,
            logdir=None,
            saver=None,
            global_step=gs,
            summary_op=None,
            save_model_secs=0,
            #local_init_op=local_init_op,
            #ready_for_local_init_op=None,
            summary_writer=None)
        conf = get_default_sess_config()
        sess = self.sv.prepare_or_wait_for_session(
            master=self.server.target,
            config=conf,
            start_standard_services=False)

        self.sess = sess
        if self.is_chief:
            print([k.name for k in tf.global_variables()])
            sess.run(global_init_op)
            logger.info("Global variables initialized.")
        #sess.run(local_init_op)
        #if self.is_chief:
            #self.config.session_init.init(self.sess)
        #self.sess.graph.finalize()
    #else:
        #logger.info("Worker {} waiting for chief".format(self.task_index))
        #self.sess = tf.train.WorkerSessionCreator(master=self.server.target).create_session()
        #logger.info("Worker wait finished")
        #self.sess.run(local_init_op)
        #logger.info("local init op runned")
        logger.info("Running post init op...")
        sess.run(self.post_init_op)
        logger.info("Post init op finished.")

        self._monitored_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=None)
        #self._monitored_sess = self.sv
        hooks = self._callbacks.get_hooks()
        self.hooked_sess = HookedSession(self.sess, hooks)

    def add_sync_queues_and_barrier(self, name_prefix, enqueue_after_list):
        """Adds ops to enqueue on all worker queues.

        Args:
            name_prefix: prefixed for the shared_name of ops.
            enqueue_after_list: control dependency from ops.

        Returns:
            an op that should be used as control dependency before starting next step.
        """
        self.sync_queue_counter += 1
        num_workers = self.cluster.num_tasks('worker')
        with tf.device(self.sync_queue_devices[self.sync_queue_counter % len(self.sync_queue_devices)]):
            sync_queues = [
                tf.FIFOQueue(num_workers, [tf.bool], shapes=[[]],
                             shared_name='%s%s' % (name_prefix, i))
                for i in range(num_workers)]
            queue_ops = []
            # For each other worker, add an entry in a queue, signaling that it can
            # finish this step.
            token = tf.constant(False)
            with tf.control_dependencies(enqueue_after_list):
                for i, q in enumerate(sync_queues):
                    if i == self.task_index:
                        queue_ops.append(tf.no_op())
                    else:
                        queue_ops.append(q.enqueue(token))

            # Drain tokens off queue for this worker, one for each other worker.
            queue_ops.append(
                sync_queues[self.task_index].dequeue_many(len(sync_queues) - 1))

            return tf.group(*queue_ops)

    def get_post_init_ops(self):
        # Copy initialized variables for variables on the parameter server
        # to the local copy of the variable.
        def strip_port(s):
            if s.endswith(':0'):
                return s[:-2]
            return s
        local_vars = tf.local_variables()
        local_var_by_name = dict([(strip_port(v.name), v) for v in local_vars])
        post_init_ops = []
        for v in tf.global_variables():
            if v.name.startswith(PS_SHADOW_VAR_PREFIX + '/'):
                prefix = strip_port(
                    v.name[len(PS_SHADOW_VAR_PREFIX + '/'):])
                for i in range(self.nr_gpu):
                    if i == 0:
                        name = prefix
                    else:
                        name = 'tower%s/%s' % (i, prefix)
                    if name in local_var_by_name:
                        copy_to = local_var_by_name[name]
                        logger.info("Post Init {} -> {}".format(v.name, copy_to.name))
                        post_init_ops.append(copy_to.assign(v.read_value()))
                    else:
                        logger.warn("Global var {} doesn't match local var".format(v.name))
        return tf.group(*post_init_ops, name='post_init_ops')
