#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import itertools, re
from six.moves import zip, range

from ..models import TowerContext
from ..utils import *
from ..utils.concurrency import LoopThread
from ..tfutils.summary import summary_moving_average
from ..tfutils.modelutils import describe_model
from ..tfutils import *

from .trainer import QueueInputTrainer

__all__ = ['AsyncMultiGPUTrainer', 'SyncMultiGPUTrainer']

class MultiGPUTrainer(QueueInputTrainer):
    """ Base class for multi-gpu training"""
    def __init__(self, config, input_queue=None, predict_tower=None):
        super(MultiGPUTrainer, self).__init__(config, input_queue, predict_tower)
        assert len(config.tower) >= 1, "MultiGPUTrainer must be used with at least one GPU."
        self.dequed_inputs = []

    @staticmethod
    def _average_grads(tower_grads):
        ret = []
        with tf.name_scope('AvgGrad'):
            for grad_and_vars in zip(*tower_grads):
                v = grad_and_vars[0][1]
                try:
                    grad = tf.add_n([x[0] for x in grad_and_vars]) / float(len(tower_grads))
                except:
                    logger.error("Error while processing gradients of {}".format(v.name))
                    raise
                ret.append((grad, v))
        return ret

    def _multi_tower_grads(self):
        logger.info("Training a model of {} tower".format(
            len(self.config.tower)))

        grad_list = []
        for idx, t in enumerate(self.config.tower):
            with tf.device('/gpu:{}'.format(t)), \
                    TowerContext('tower{}'.format(idx)) as scope:
                logger.info("Building graph for training tower {}...".format(idx))
                model_inputs = self._get_model_inputs()    # each tower dequeue from input queue
                self.dequed_inputs.append(model_inputs)

                self.model.build_graph(model_inputs)
                cost_var = self.model.get_cost() # build tower

                # TODO gate_gradienst=0 seems to be faster?
                grad_list.append(
                    self.config.optimizer.compute_gradients(cost_var, gate_gradients=0))

                if idx == 0:
                    tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost_var)
                    tf.get_variable_scope().reuse_variables()
                    # avoid repeated summary from each device
                    backup = backup_collection(SUMMARY_BACKUP_KEYS)
        restore_collection(backup)
        return grad_list

class SyncMultiGPUTrainer(MultiGPUTrainer):
    def train(self):
        self.init_session_and_coord()
        self._build_enque_thread()

        grad_list = self._multi_tower_grads()

        grads = MultiGPUTrainer._average_grads(grad_list)
        grads = self.process_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')
        describe_model()

        # [debug]: do nothing in training
        #self.train_op = self.dequed_inputs[0][0] + self.dequed_inputs[1][0]
        self.main_loop()

class AsyncMultiGPUTrainer(MultiGPUTrainer):
    def train(self):
        self.init_session_and_coord()
        self._build_enque_thread()

        grad_list = self._multi_tower_grads()
        # pretend to average the grads, in order to make async and
        # sync have consistent effective learning rate
        def scale(grads):
            with tf.name_scope('AsyncScaleGrad'):
                return [(grad / len(self.config.tower) if grad is not None else None, var)
                            for grad, var in grads]
        grad_list = map(scale, grad_list)
        grad_list = [self.process_grads(g) for g in grad_list]

        # use grad from the first tower for iteration in main thread
        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grad_list[0], get_global_step_var()),
            summary_moving_average(), name='train_op')
        describe_model()

        self._start_async_threads(grad_list)

        self.main_loop()

    def _start_async_threads(self, grad_list):
        # prepare train_op for the rest of the towers
        # itertools.count is atomic w.r.t. python threads
        self.async_step_counter = itertools.count()
        self.training_threads = []
        for k in range(1, len(self.config.tower)):
            train_op = self.config.optimizer.apply_gradients(grad_list[k])
            def f(op=train_op): # avoid late-binding
                self.sess.run([op])
                next(self.async_step_counter)
            th = LoopThread(f)
            th.pause()
            th.start()
            self.training_threads.append(th)
        self.async_running = False

    def run_step(self):
        if not self.async_running:
            self.async_running = True
            for th in self.training_threads: # resume all threads
                th.resume()
        next(self.async_step_counter)
        super(AsyncMultiGPUTrainer, self).run_step()

    def _trigger_epoch(self):
        self.async_running = False
        for th in self.training_threads:
            th.pause()
        try:
            async_step_total_cnt = int(re.findall(
                '[0-9]+', self.async_step_counter.__str__())[0])
            self.write_scalar_summary(
                    'async_global_step', async_step_total_cnt)
        except:
            pass
        super(AsyncMultiGPUTrainer, self)._trigger_epoch()
