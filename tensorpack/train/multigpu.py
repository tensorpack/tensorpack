#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import itertools
import re
from six.moves import zip, range

from ..utils import logger
from ..utils.develop import log_deprecated
from ..utils.naming import SUMMARY_BACKUP_KEYS
from ..utils.concurrency import LoopThread
from ..tfutils.tower import TowerContext
from ..tfutils.collection import backup_collection, restore_collection
from ..tfutils.gradproc import FilterNoneGrad, ScaleGradient

from .base import Trainer
from .feedfree import SingleCostFeedfreeTrainer
from .input_data import QueueInput

__all__ = ['SyncMultiGPUTrainer', 'AsyncMultiGPUTrainer']


class MultiGPUTrainer(Trainer):
    """ Base class for multi-gpu training"""
    @staticmethod
    def _multi_tower_grads(towers, get_tower_grad_func):
        """ ret[i] is a lists of (grad,var) tuple for tower i"""
        logger.info("Training a model of {} tower".format(len(towers)))

        grad_list = []
        global_scope = tf.get_variable_scope()
        for idx, t in enumerate(towers):
            with tf.device('/gpu:{}'.format(t)), \
                    tf.variable_scope(global_scope, reuse=idx > 0), \
                    TowerContext('tower{}'.format(idx)):
                logger.info("Building graph for training tower {}...".format(idx))

                grad_list.append(get_tower_grad_func())

                if idx == 0:
                    # avoid repeated summary from each device
                    backup = backup_collection(SUMMARY_BACKUP_KEYS)
        restore_collection(backup)
        return grad_list

    @staticmethod
    def _multi_tower_costs(towers, get_tower_cost_func):
        logger.info("Training a model of {} tower".format(len(towers)))

        cost_list = []
        global_scope = tf.get_variable_scope()
        for idx, t in enumerate(towers):
            with tf.device('/gpu:{}'.format(t)), \
                    tf.variable_scope(global_scope, reuse=idx > 0), \
                    TowerContext('tower{}'.format(idx)):
                logger.info("Building graph for training tower {}...".format(idx))

                cost_list.append(get_tower_cost_func())

                if idx == 0:
                    # avoid repeated summary from each device
                    backup = backup_collection(SUMMARY_BACKUP_KEYS)
        restore_collection(backup)
        return cost_list


class SyncMultiGPUTrainer(MultiGPUTrainer,
                          SingleCostFeedfreeTrainer):
    """
    A multi-tower multi-GPU trainer which synchronoizes the gradients computed
    from each tower and averages them.
    """

    def __init__(self, config, input_queue=None,
                 average_cost=False,
                 predict_tower=None):
        """
        Args:
            config, input_queue: same as in :class:`QueueInputTrainer`.
            average_cost (bool): average the cost (instead of gradients) from
                each tower and did backprop only once. This option should make no
                difference mathematically, but may affect speed.
        """
        if config.dataflow is not None:
            # use queueinput by default. May need to avoid this in the future (when more input type is available)
            self._input_method = QueueInput(config.dataflow, input_queue)
        else:
            self._input_method = config.data
            assert isinstance(self._input_method, QueueInput)

        if predict_tower is not None:
            log_deprecated("Argument `predict_tower` in trainer", "Use TrainConfig(predict_tower=...) instead!")
            config.predict_tower = predict_tower

        super(SyncMultiGPUTrainer, self).__init__(config)
        assert len(config.tower) >= 1, "MultiGPUTrainer must be used with at least one GPU."
        assert tf.test.is_gpu_available()
        self.average_cost = average_cost

    @staticmethod
    def _average_grads(tower_grads):
        if len(tower_grads) == 1:
            return tower_grads[0]
        ret = []
        with tf.name_scope('AvgGrad'):
            for grad_and_vars in zip(*tower_grads):
                v = grad_and_vars[0][1]
                all_grad = [k[0] for k in grad_and_vars]

                nones = list(set(all_grad))
                if None in nones and len(nones) != 1:
                    raise RuntimeError("Gradient w.r.t {} is None in some but not all towers!".format(v.name))
                elif nones[0] is None:
                    logger.warn("No Gradient w.r.t {}".format(v.op.name))
                    continue
                try:
                    grad = tf.add_n(all_grad) / float(len(tower_grads))
                except:
                    logger.error("Error while processing gradients of {}".format(v.name))
                    raise
                ret.append((grad, v))
        return ret

    def _setup(self):
        super(SyncMultiGPUTrainer, self)._setup()
        if not self.average_cost:
            grad_list = MultiGPUTrainer._multi_tower_grads(
                self.config.tower, lambda: self._get_cost_and_grad()[1])

            # debug tower performance (without update):
            # ops = [k[0] for k in grad_list[1]] + [k[0] for k in grad_list[0]]
            # self.train_op = tf.group(*ops)
            # return

            grads = SyncMultiGPUTrainer._average_grads(grad_list)
            # grads = grad_list[0]
        else:
            def get_cost():
                self.build_train_tower()
                return self.model.get_cost()

            cost_list = MultiGPUTrainer._multi_tower_costs(
                self.config.tower, get_cost)
            cost = tf.multiply(tf.add_n(cost_list), 1.0 / len(cost_list),
                               name='averaged_cost')

            opt = self.config.optimizer
            grads = opt.compute_gradients(
                cost,
                gate_gradients=tf.train.Optimizer.GATE_NONE,
                colocate_gradients_with_ops=True)
        self.train_op = self.config.optimizer.apply_gradients(grads, name='min_op')


class AsyncMultiGPUTrainer(MultiGPUTrainer,
                           SingleCostFeedfreeTrainer):
    """
    A multi-tower multi-GPU trainer where each tower independently
    asynchronously updates the model without locking.
    """

    def __init__(self, config,
                 input_queue=None,
                 scale_gradient=True,
                 predict_tower=None):
        """
        Args:
            config, input_queue: same as in :class:`QueueInputTrainer`.
            scale_gradient (bool): if True, will scale each gradient by
                ``1.0/nr_tower``, to make Async and Sync Trainer have the same
                effective learning rate.
        """
        if config.dataflow is not None:
            self._input_method = QueueInput(config.dataflow, input_queue)
        else:
            self._input_method = config.data
            assert isinstance(self._input_method, QueueInput)
        super(AsyncMultiGPUTrainer, self).__init__(config)

        if predict_tower is not None:
            log_deprecated("Argument `predict_tower` in trainer", "Use TrainConfig.predict_tower instead!")
            config.predict_tower = predict_tower

        self._scale_gradient = scale_gradient
        assert tf.test.is_gpu_available()

    def _setup(self):
        super(AsyncMultiGPUTrainer, self)._setup()
        grad_list = MultiGPUTrainer._multi_tower_grads(
            self.config.tower, lambda: self._get_cost_and_grad()[1])
        grad_list = [FilterNoneGrad().process(gv) for gv in grad_list]
        if self._scale_gradient and self.config.nr_tower > 1:
            # pretend to average the grads, in order to make async and
            # sync have consistent effective learning rate
            gradproc = ScaleGradient(('.*', 1.0 / self.config.nr_tower), log=False)
            grad_list = [gradproc.process(gv) for gv in grad_list]

        # use grad from the first tower for iteration in main thread
        self.train_op = self.config.optimizer.apply_gradients(grad_list[0], name='min_op')

        self._start_async_threads(grad_list)

    def _start_async_threads(self, grad_list):
        # prepare train_op for the rest of the towers
        # itertools.count is atomic w.r.t. python threads
        self.async_step_counter = itertools.count()
        self.training_threads = []
        for k in range(1, self.config.nr_tower):
            train_op = self.config.optimizer.apply_gradients(grad_list[k])

            def f(op=train_op):  # avoid late-binding
                self.sess.run([op])
                next(self.async_step_counter)   # atomic due to GIL
            th = LoopThread(f)
            th.name = "AsyncLoopThread-{}".format(k)
            th.pause()
            th.start()
            self.training_threads.append(th)
        self.async_running = False

    def run_step(self):
        if not self.async_running:
            self.async_running = True
            for th in self.training_threads:  # resume all threads
                th.resume()
        next(self.async_step_counter)
        return super(AsyncMultiGPUTrainer, self).run_step()

    def _trigger_epoch(self):
        self.async_running = False
        for th in self.training_threads:
            th.pause()
        try:
            if self.config.nr_tower > 1:
                async_step_total_cnt = int(re.findall(
                    '[0-9]+', self.async_step_counter.__str__())[0])
                self.monitors.put(
                    'async_global_step', async_step_total_cnt)
        except:
            logger.exception("Cannot log async_global_step")
        super(AsyncMultiGPUTrainer, self)._trigger_epoch()
