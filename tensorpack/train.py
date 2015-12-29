#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count
import argparse

from utils import *
from utils.concurrency import EnqueueThread,coordinator_guard
from utils.summary import summary_moving_average
from utils.modelutils import describe_model
from utils.sessinit import NewSession
from utils import logger
from dataflow import DataFlow

def prepare():
    global_step_var = tf.Variable(
        0, trainable=False, name=GLOBAL_STEP_OP_NAME)

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
        config: a tensorpack config dictionary
    """
    dataset = config['dataset']
    assert isinstance(dataset, DataFlow), dataset.__class__

    # a tf.train.Optimizer instance
    optimizer = config['optimizer']
    assert isinstance(optimizer, tf.train.Optimizer), optimizer.__class__

    # a list of Callback instance
    callbacks = config['callback']

    # a tf.ConfigProto instance
    sess_config = config.get('session_config', None)
    assert isinstance(sess_config, tf.ConfigProto), sess_config.__class__

    sess_init = config.get('session_init', NewSession())

    # input/output variables
    input_vars = config['inputs']
    input_queue = config['input_queue']
    get_model_func = config['get_model_func']

    step_per_epoch = int(config['step_per_epoch'])
    max_epoch = int(config['max_epoch'])
    assert step_per_epoch > 0 and max_epoch > 0

    enqueue_op = input_queue.enqueue(tuple(input_vars))
    model_inputs = input_queue.dequeue()
    # set dequeue shape
    for qv, v in zip(model_inputs, input_vars):
        qv.set_shape(v.get_shape())
    output_vars, cost_var = get_model_func(model_inputs, is_training=True)

    # build graph
    tf.add_to_collection(FORWARD_FUNC_KEY, get_model_func)
    for v in input_vars:
        tf.add_to_collection(INPUT_VARS_KEY, v)
    for v in output_vars:
        tf.add_to_collection(OUTPUT_VARS_KEY, v)
    describe_model()

    train_op = get_train_op(optimizer, cost_var)

    sess = tf.Session(config=sess_config)
    sess_init.init(sess)

    # start training:
    coord = tf.train.Coordinator()
    # a thread that keeps filling the queue
    input_th = EnqueueThread(sess, coord, enqueue_op, dataset)
    model_th = tf.train.start_queue_runners(
        sess=sess, coord=coord, daemon=True, start=True)
    input_th.start()

    with sess.as_default(), \
            coordinator_guard(sess, coord):
        callbacks.before_train()
        for epoch in xrange(1, max_epoch):
            with timed_operation('epoch {}'.format(epoch)):
                for step in xrange(step_per_epoch):
                    if coord.should_stop():
                        return
                    fetches = [train_op, cost_var] + output_vars + model_inputs
                    results = sess.run(fetches)
                    cost = results[1]
                    outputs = results[2:2 + len(output_vars)]
                    inputs = results[-len(model_inputs):]
                    callbacks.trigger_step(inputs, outputs, cost)

                # note that summary_op will take a data from the queue.
                callbacks.trigger_epoch()

