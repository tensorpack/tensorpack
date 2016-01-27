#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count
import copy
import argparse

import tqdm
from utils import *
from utils.concurrency import EnqueueThread
from callbacks import *
from utils.summary import summary_moving_average
from utils.modelutils import describe_model
from utils import logger
from dataflow import DataFlow

class TrainConfig(object):
    """ config for training"""
    def __init__(self, **kwargs):
        """
        Args:
            dataset: the dataset to train. a tensorpack.dataflow.DataFlow instance.
            optimizer: a tf.train.Optimizer instance defining the optimizer
                for trainig. default to an AdamOptimizer
            callbacks: a tensorpack.utils.callback.Callbacks instance. Define
                the callbacks to perform during training. has to contain a
                SummaryWriter and a PeriodicSaver
            session_config: a tf.ConfigProto instance to instantiate the
                session. default to a session running 1 GPU.
            session_init: a tensorpack.utils.sessinit.SessionInit instance to
                initialize variables of a session. default to a new session.
            inputs: a list of input variables. must match what is returned by
                the dataset
            input_queue: the queue used for input. default to a FIFO queue
                with capacity 5
            get_model_func: a function taking `inputs` and `is_training`, and
                return the cost to minimize
            batched_model_input: boolean. If yes, `get_model_func` expected batched
                input in training. Otherwise, expect single data point in
                training, so that you may do pre-processing and batch them
                later with batch ops. It's suggested that you do all
                preprocessing in dataset as that is usually faster.
            step_per_epoch: the number of steps (parameter updates) to perform
                in each epoch. default to dataset.size()
            max_epoch: maximum number of epoch to run training. default to 100
            nr_tower: int. number of towers. default to 1.
        """
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__
        self.dataset = kwargs.pop('dataset')
        assert_type(self.dataset, DataFlow)
        self.optimizer = kwargs.pop('optimizer', tf.train.AdamOptimizer())
        assert_type(self.optimizer, tf.train.Optimizer)
        self.callbacks = kwargs.pop('callbacks')
        assert_type(self.callbacks, Callbacks)
        self.session_config = kwargs.pop('session_config', get_default_sess_config())
        assert_type(self.session_config, tf.ConfigProto)
        self.session_init = kwargs.pop('session_init', NewSession())
        assert_type(self.session_init, SessionInit)
        self.inputs = kwargs.pop('inputs')
        [assert_type(i, tf.Tensor) for i in self.inputs]
        self.input_queue = kwargs.pop(
            'input_queue', tf.FIFOQueue(5, [x.dtype for x in self.inputs], name='input_queue'))
        assert_type(self.input_queue, tf.QueueBase)
        assert self.input_queue.dtypes == [x.dtype for x in self.inputs]
        self.get_model_func = kwargs.pop('get_model_func')
        self.batched_model_input = kwargs.pop('batched_model_input', True)
        self.step_per_epoch = int(kwargs.pop('step_per_epoch', self.dataset.size()))
        self.max_epoch = int(kwargs.pop('max_epoch', 100))
        assert self.step_per_epoch > 0 and self.max_epoch > 0
        self.nr_tower = int(kwargs.pop('nr_tower', 1))
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grad = tf.add_n([x[0] for x in grad_and_vars]) / float(len(tower_grads))
        v = grad_and_vars[0][1]
        average_grads.append((grad, v))
    return average_grads

def summary_grads(grads):
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

def check_grads(grads):
    for grad, var in grads:
        tf.Assert(tf.reduce_all(tf.is_finite(var)), [var])

def start_train(config):
    """
    Start training with the given config
    Args:
        config: a TrainConfig instance
    """
    input_vars = config.inputs
    input_queue = config.input_queue
    callbacks = config.callbacks

    tf.add_to_collection(FORWARD_FUNC_KEY, config.get_model_func)
    for v in input_vars:
        tf.add_to_collection(INPUT_VARS_KEY, v)

    def get_model_inputs():
        model_inputs = input_queue.dequeue()
        if isinstance(model_inputs, tf.Tensor):
            model_inputs = [model_inputs]
        for qv, v in zip(model_inputs, input_vars):
            if config.batched_model_input:
                qv.set_shape(v.get_shape())
            else:
                qv.set_shape(v.get_shape().as_list()[1:])
        return model_inputs

    if config.batched_model_input:
        enqueue_op = input_queue.enqueue(input_vars)
    else:
        enqueue_op = input_queue.enqueue_many(input_vars)

    # get gradients to update:
    logger.info("Training a model of {} tower".format(config.nr_tower))
    if config.nr_tower > 1:
        coll_keys = [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_VARS_KEY]
        kept_summaries = {}
        grads = []
        for i in range(config.nr_tower):
            with tf.device('/gpu:{}'.format(i)), \
                    tf.name_scope('tower{}'.format(i)) as scope:
                model_inputs = get_model_inputs()
                cost_var = config.get_model_func(model_inputs, is_training=True)
                grads.append(
                    config.optimizer.compute_gradients(cost_var))

                if i == 0:
                    tf.get_variable_scope().reuse_variables()
                    for k in coll_keys:
                        kept_summaries[k] = copy.copy(tf.get_collection(k))
        for k in coll_keys:  # avoid repeating summary on multiple devices
            del tf.get_collection(k)[:]
            tf.get_collection(k).extend(kept_summaries[k])
        grads = average_gradients(grads)
    else:
        model_inputs = get_model_inputs()
        cost_var = config.get_model_func(model_inputs, is_training=True)
        grads = config.optimizer.compute_gradients(cost_var)
    summary_grads(grads)
    check_grads(grads)
    avg_maintain_op = summary_moving_average(cost_var)

    train_op = tf.group(
        config.optimizer.apply_gradients(grads, get_global_step_var()),
        avg_maintain_op)

    describe_model()
    sess = tf.Session(config=config.session_config)
    config.session_init.init(sess)

    # start training:
    coord = tf.train.Coordinator()
    # a thread that keeps filling the queue
    model_th = tf.train.start_queue_runners(
        sess=sess, coord=coord, daemon=True, start=True)
    input_th = EnqueueThread(sess, coord, enqueue_op, config.dataset, input_queue)
    input_th.start()

    with sess.as_default():
        try:
            logger.info("Start training with global_step={}".format(get_global_step()))
            callbacks.before_train()
            for epoch in xrange(1, config.max_epoch):
                with timed_operation('epoch {}'.format(epoch)):
                    for step in tqdm.trange(
                            config.step_per_epoch, leave=True, mininterval=0.2):
                        if coord.should_stop():
                            return
                        sess.run([train_op])    # faster since train_op return None
                        callbacks.trigger_step()

                    # note that summary_op will take a data from the queue.
                    callbacks.trigger_epoch()
        except (KeyboardInterrupt, Exception):
            raise
        finally:
            coord.request_stop()
            input_queue.close(cancel_pending_enqueues=True)
            callbacks.after_train()
            sess.close()

