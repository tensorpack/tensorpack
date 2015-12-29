#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: infer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count
import argparse
import numpy as np

from utils import *
from utils.modelutils import describe_model, restore_params
from utils import logger
from dataflow import DataFlow

def start_infer(config):
    """
    Args:
        config: a tensorpack config dictionary
    """
    dataset = config['dataset']
    assert isinstance(dataset, DataFlow), dataset.__class__

    # a tf.ConfigProto instance
    sess_config = config.get('session_config', None)
    assert isinstance(sess_config, tf.ConfigProto), sess_config.__class__

    # TODO callback should have trigger_step and trigger_end?
    callback = config['callback']

    # restore saved params
    params = config.get('restore_params', {})

    # input/output variables
    input_vars = config['inputs']
    get_model_func = config['get_model_func']

    output_vars, cost_var = get_model_func(input_vars, is_training=False)

    # build graph
    G = tf.get_default_graph()
    G.add_to_collection(FORWARD_FUNC_KEY, get_model_func)
    for v in input_vars:
        G.add_to_collection(INPUT_VARS_KEY, v)
    for v in output_vars:
        G.add_to_collection(OUTPUT_VARS_KEY, v)
    describe_model()

    sess = tf.Session(config=sess_config)
    sess.run(tf.initialize_all_variables())

    restore_params(sess, params)

    with sess.as_default():
        with timed_operation('running one batch'):
            for dp in dataset.get_data():
                feed = dict(zip(input_vars, dp))
                fetches = [cost_var] + output_vars
                results = sess.run(fetches, feed_dict=feed)
                cost = results[0]
                outputs = results[1:]
                prob = outputs[0]
                callback(dp, outputs, cost)

def main(get_config_func):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with tf.Graph().as_default():
        config = get_config_func()
        start_infer(config)
