#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import itertools, re
from six.moves import zip, range

from ..utils import logger
from ..utils.naming import *
from ..utils.concurrency import LoopThread
from ..tfutils.summary import summary_moving_average, add_moving_summary
from ..tfutils import (backup_collection, restore_collection,
        get_global_step_var, TowerContext)
from ..tfutils.gradproc import apply_grad_processors, ScaleGradient

from .trainer import FeedlessTrainer, SingleCostFeedlessTrainer, MultiPredictorTowerTrainer
from .queue import QueueInputTrainer, QueueInputTrainerBase

__all__ = ['AsyncMultiGPUTrainer', 'SyncMultiGPUTrainer']

class MultiGPUTrainer(FeedlessTrainer):
    """ Base class for multi-gpu training"""
    @staticmethod
    def _multi_tower_grads(towers, get_tower_grad_func):
        logger.info("Training a model of {} tower".format(len(towers)))

        grad_list = []
        global_scope = tf.get_variable_scope()
        for idx, t in enumerate(towers):
            with tf.device('/gpu:{}'.format(t)), \
                    tf.variable_scope(global_scope, reuse=idx > 0), \
                    TowerContext('tower{}'.format(idx)) as scope:
                logger.info("Building graph for training tower {}...".format(idx))

                grad_list.append(get_tower_grad_func())

                if idx == 0:
                    # avoid repeated summary from each device
                    backup = backup_collection(SUMMARY_BACKUP_KEYS)
        restore_collection(backup)
        return grad_list

class SyncMultiGPUTrainer(QueueInputTrainerBase,
        MultiGPUTrainer,
        SingleCostFeedlessTrainer,
        MultiPredictorTowerTrainer):
    def __init__(self, config, input_queue=None, predict_tower=None):
        assert len(config.tower) >= 1, "MultiGPUTrainer must be used with at least one GPU."
        super(SyncMultiGPUTrainer, self).__init__(config)
        self._setup_predictor_factory(predict_tower)
        self._build_enque_thread(input_queue)

    @staticmethod
    def _average_grads(tower_grads):
        ret = []
        with tf.name_scope('AvgGrad'):
            for grad_and_vars in zip(*tower_grads):
                v = grad_and_vars[0][1]
                all_grad = [k[0] for k in grad_and_vars]

                nones = list(set(all_grad))
                if None in nones and len(nones) != 1:
                    raise RuntimeError("Gradient w.r.t {} is None in some but not all towers!".format(v.name))
                elif nones[0] is None:
                    logger.warn("No Gradient w.r.t {}".format(var.op.name))
                    continue
                try:
                    grad = tf.add_n(all_grad) / float(len(tower_grads))
                except:
                    logger.error("Error while processing gradients of {}".format(v.name))
                    raise
                ret.append((grad, v))
        return ret

    def _setup(self):
        grad_list = MultiGPUTrainer._multi_tower_grads(
                self.config.tower, lambda: self._get_cost_and_grad()[1])
        grads = SyncMultiGPUTrainer._average_grads(grad_list)
        grads = apply_grad_processors(grads, self.model.get_gradient_processor())

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')

    def run_step(self):
        self.sess.run(self.train_op)

class AsyncMultiGPUTrainer(QueueInputTrainerBase,
        MultiGPUTrainer,
        SingleCostFeedlessTrainer,
        MultiPredictorTowerTrainer):
    def __init__(self, config, input_queue=None, predict_tower=None):
        super(AsyncMultiGPUTrainer, self).__init__(config)
        self._setup_predictor_factory(predict_tower)
        self._build_enque_thread(input_queue)

    def _setup(self):
        grad_list = MultiGPUTrainer._multi_tower_grads(
                self.config.tower, lambda: self._get_cost_and_grad()[1])
        gradprocs = self.model.get_gradient_processor()
        # pretend to average the grads, in order to make async and
        # sync have consistent effective learning rate
        if self.config.nr_tower > 1:
            gradprocs.insert(0, ScaleGradient(('.*', 1.0 / self.config.nr_tower), log=False))
        grad_list = [apply_grad_processors(g, gradprocs) for g in grad_list]

        # use grad from the first tower for iteration in main thread
        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(
                grad_list[0], get_global_step_var()),
            summary_moving_average(), name='train_op')

        self._start_async_threads(grad_list)

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
        self.sess.run(self.train_op)

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
            logger.exception("Cannot log async_global_step")
        super(AsyncMultiGPUTrainer, self)._trigger_epoch()
