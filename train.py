#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from utils import *
from utils.concurrency import *
from utils.callback import *
from utils.summary import *
from dataflow import DataFlow
from itertools import count
import argparse

def prepare():
    is_training = tf.constant(True, name=IS_TRAINING_OP_NAME)
    #keep_prob = tf.placeholder(
        #tf.float32, shape=tuple(), name=DROPOUT_PROB_OP_NAME)
    global_step_var = tf.Variable(
        0, trainable=False, name=GLOBAL_STEP_OP_NAME)


def start_train(config):
    """
    Start training with the given config
    Args:
        config: a tensorpack config dictionary
    """
    dataset_train = config['dataset_train']
    assert isinstance(dataset_train, DataFlow), dataset_train.__class__

    # a tf.train.Optimizer instance
    optimizer = config['optimizer']
    assert isinstance(optimizer, tf.train.Optimizer), optimizer.__class__

    # a list of Callback instance
    callbacks = config['callback']

    # a tf.ConfigProto instance
    sess_config = config.get('session_config', None)
    assert isinstance(sess_config, tf.ConfigProto), sess_config.__class__

    # a list of input/output variables
    input_vars = config['inputs']
    input_queue = config['input_queue']
    get_model_func = config['get_model_func']

    max_epoch = int(config['max_epoch'])

    enqueue_op = input_queue.enqueue(tuple(input_vars))
    model_inputs = input_queue.dequeue()
    for qv, v in zip(model_inputs, input_vars):
        qv.set_shape(v.get_shape())
    output_vars, cost_var = get_model_func(model_inputs)

    # build graph
    G = tf.get_default_graph()
    G.add_to_collection(FORWARD_FUNC_KEY, get_model_func)
    for v in input_vars:
        G.add_to_collection(INPUT_VARS_KEY, v)
    for v in output_vars:
        G.add_to_collection(OUTPUT_VARS_KEY, v)
    describe_model()

    global_step_var = G.get_tensor_by_name(GLOBAL_STEP_VAR_NAME)

    avg_maintain_op = summary_moving_average(cost_var)

    # maintain average in each step
    with tf.control_dependencies([avg_maintain_op]):
        grads = optimizer.compute_gradients(cost_var)

    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    train_op = optimizer.apply_gradients(grads, global_step_var)

    sess = tf.Session(config=sess_config)
    sess.run(tf.initialize_all_variables())

    # start training:
    coord = tf.train.Coordinator()
    # a thread that keeps filling the queue
    th = EnqueueThread(sess, coord, enqueue_op, dataset_train)
    with sess.as_default(), \
            coordinator_context(
                sess, coord, th, input_queue):
        callbacks.before_train()
        for epoch in xrange(1, max_epoch):
            with timed_operation('epoch {}'.format(epoch)):
                for step in xrange(dataset_train.size()):
                    # TODO eval dequeue to get dp
                    fetches = [train_op, cost_var] + output_vars
                    feed = {IS_TRAINING_VAR_NAME: True}
                    results = sess.run(fetches, feed_dict=feed)
                    cost = results[1]
                    outputs = results[2:]
                    # TODO trigger_step
                # note that summary_op will take a data from the queue.
                callbacks.trigger_epoch()
    sess.close()

def main(get_config_func):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with tf.Graph().as_default():
        prepare()
        config = get_config_func()
        start_train(config)
