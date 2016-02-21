#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import copy
import re

from .base import Trainer
from ..dataflow.common import RepeatedData
from ..utils import *
from ..utils.concurrency import EnqueueThread
from ..utils.summary import summary_moving_average

__all__ = ['SimpleTrainer', 'QueueInputTrainer']

def summary_grads(grads):
    for grad, var in grads:
        if grad:
            # TODO also summary RMS and print
            tf.histogram_summary(var.op.name + '/gradients', grad)

def check_grads(grads):
    for grad, var in grads:
        assert grad is not None, "Grad is None for variable {}".format(var.name)
        tf.Assert(tf.reduce_all(tf.is_finite(var)), [var])

def scale_grads(grads, multiplier):
    ret = []
    for grad, var in grads:
        varname = var.name
        for regex, val in multiplier:
            if re.search(regex, varname):
                logger.info("Apply lr multiplier {} for {}".format(val, varname))
                ret.append((grad * val, var))
                break
        else:
            ret.append((grad, var))
    return ret


class SimpleTrainer(Trainer):
    def run_step(self):
        data = next(self.data_producer)
        feed = dict(zip(self.input_vars, data))
        self.sess.run([self.train_op], feed_dict=feed)    # faster since train_op return None

    def train(self):
        model = self.config.model
        input_vars = model.get_input_vars()
        self.input_vars = input_vars
        cost_var = model.get_cost(input_vars, is_training=True)
        avg_maintain_op = summary_moving_average(cost_var)

        grads = self.config.optimizer.compute_gradients(cost_var)
        check_grads(grads)
        grads = scale_grads(grads, model.get_lr_multiplier())
        summary_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            avg_maintain_op)

        self.init_session_and_coord()
        # create an infinte data producer
        self.data_producer = RepeatedData(self.config.dataset, -1).get_data()
        self.main_loop()

    def _trigger_epoch(self):
        if self.summary_op is not None:
            data = next(self.data_producer)
            feed = dict(zip(self.input_vars, data))
            summary_str = self.summary_op.eval(feed_dict=feed)
            self._process_summary(summary_str)


class QueueInputTrainer(Trainer):
    """
    Trainer which builds a queue for input.
    Support multi GPU.
    """

    @staticmethod
    def _average_grads(tower_grads):
        ret = []
        for grad_and_vars in zip(*tower_grads):
            grad = tf.add_n([x[0] for x in grad_and_vars]) / float(len(tower_grads))
            v = grad_and_vars[0][1]
            ret.append((grad, v))
        return ret

    def train(self):
        model = self.config.model
        input_vars = model.get_input_vars()
        input_queue = model.get_input_queue()

        enqueue_op = input_queue.enqueue(input_vars)
        def get_model_inputs():
            model_inputs = input_queue.dequeue()
            if isinstance(model_inputs, tf.Tensor): # only one input
                model_inputs = [model_inputs]
            for qv, v in zip(model_inputs, input_vars):
                qv.set_shape(v.get_shape())
            return model_inputs

        # get gradients to update:
        if self.config.nr_tower > 1:
            logger.info("Training a model of {} tower".format(self.config.nr_tower))
            # to avoid repeated summary from each device
            coll_keys = [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_VARS_KEY]
            kept_summaries = {}
            grad_list = []
            for i in range(self.config.nr_tower):
                with tf.device('/gpu:{}'.format(i)), \
                        tf.name_scope('tower{}'.format(i)) as scope:
                    model_inputs = get_model_inputs()
                    cost_var = model.get_cost(model_inputs, is_training=True)
                    grad_list.append(
                        self.config.optimizer.compute_gradients(cost_var))

                    if i == 0:
                        tf.get_variable_scope().reuse_variables()
                        for k in coll_keys:
                            kept_summaries[k] = copy.copy(tf.get_collection(k))
            for k in coll_keys:
                del tf.get_collection(k)[:]
                tf.get_collection(k).extend(kept_summaries[k])
            grads = QueueInputTrainer._average_grads(grad_list)
        else:
            model_inputs = get_model_inputs()
            cost_var = model.get_cost(model_inputs, is_training=True)
            grads = self.config.optimizer.compute_gradients(cost_var)
        avg_maintain_op = summary_moving_average(cost_var)  # TODO(multigpu) average the cost from each device?

        check_grads(grads)
        grads = scale_grads(grads, model.get_lr_multiplier())
        summary_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            avg_maintain_op)

        self.init_session_and_coord()

        # create a thread that keeps filling the queue
        input_th = EnqueueThread(self.sess, self.coord, enqueue_op, self.config.dataset, input_queue)
        input_th.start()
        self.main_loop()

    def run_step(self):
        self.sess.run([self.train_op])    # faster since train_op return None

    def _trigger_epoch(self):
        # note that summary_op will take a data from the queue
        if self.summary_op is not None:
            summary_str = self.summary_op.eval()
            self._process_summary(summary_str)


def start_train(config):
    tr = SimpleTrainer(config)
    tr.train()
