#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count
import argparse

from utils import *
from utils.concurrency import EnqueueThread,coordinator_guard
from utils.callback import Callbacks
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
            get_model_func: a function taking `inputs` and `is_training` and
                return a tuple of output list as well as the cost to minimize
            batched_model_input: boolean. If yes, `get_model_func` expected batched
                input in training. Otherwise, expect single data point in
                training, so that you may do pre-processing and batch them
                later with batch ops. It's suggested that you do all
                preprocessing in dataset as that is usually faster.
            step_per_epoch: the number of steps (parameter updates) to perform
                in each epoch. default to dataset.size()
            max_epoch: maximum number of epoch to run training. default to 100
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
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

def get_train_op(optimizer, cost_var):
    global_step_var = tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)

    avg_maintain_op = summary_moving_average(cost_var)

    # maintain average in each step
    with tf.control_dependencies([avg_maintain_op]):
        grads = optimizer.compute_gradients(cost_var)

    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    return optimizer.apply_gradients(grads, global_step_var)

def start_train(config):
    """
    Start training with the given config
    Args:
        config: a TrainConfig instance
    """
    input_vars = config.inputs
    input_queue = config.input_queue
    callbacks = config.callbacks

    if config.batched_model_input:
        enqueue_op = input_queue.enqueue(input_vars)
        model_inputs = input_queue.dequeue()
        for qv, v in zip(model_inputs, input_vars):
            qv.set_shape(v.get_shape())
    else:
        enqueue_op = input_queue.enqueue_many(input_vars)
        model_inputs = input_queue.dequeue()
        for qv, v in zip(model_inputs, input_vars):
            qv.set_shape(v.get_shape().as_list()[1:])
    output_vars, cost_var = config.get_model_func(model_inputs, is_training=True)

    # build graph
    tf.add_to_collection(FORWARD_FUNC_KEY, config.get_model_func)
    for v in input_vars:
        tf.add_to_collection(INPUT_VARS_KEY, v)
    for v in output_vars:
        tf.add_to_collection(OUTPUT_VARS_KEY, v)
    describe_model()

    train_op = get_train_op(config.optimizer, cost_var)

    sess = tf.Session(config=config.session_config)
    config.session_init.init(sess)

    # start training:
    coord = tf.train.Coordinator()
    # a thread that keeps filling the queue
    input_th = EnqueueThread(sess, coord, enqueue_op, config.dataset, input_queue)
    model_th = tf.train.start_queue_runners(
        sess=sess, coord=coord, daemon=True, start=True)
    input_th.start()

    with sess.as_default(), \
            coordinator_guard(sess, coord):
        callbacks.before_train()
        for epoch in xrange(1, config.max_epoch):
            with timed_operation('epoch {}'.format(epoch)):
                for step in xrange(config.step_per_epoch):
                    if coord.should_stop():
                        return
                    # TODO if no one uses trigger_step, train_op can be
                    # faster, see: https://github.com/soumith/convnet-benchmarks/pull/67/files
                    fetches = [train_op, cost_var] + output_vars + model_inputs
                    results = sess.run(fetches)
                    cost = results[1]
                    outputs = results[2:2 + len(output_vars)]
                    inputs = results[-len(model_inputs):]
                    callbacks.trigger_step(inputs, outputs, cost)

                # note that summary_op will take a data from the queue.
                callbacks.trigger_epoch()
