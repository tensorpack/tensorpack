#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: infer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count
import argparse
import numpy as np

from utils import *
from utils.modelutils import describe_model
from utils import logger
from dataflow import DataFlow

def get_predict_func(config):
    """
    Args:
        config: a tensorpack config dictionary
    Returns:
        a function that takes a list of inputs to run the model
    """
    sess_config = config.get('session_config', None)
    if sess_config is None:
        sess_config = get_default_sess_config()
    assert isinstance(sess_config, tf.ConfigProto), sess_config.__class__

    sess_init = config['session_init']

    # input/output variables
    input_vars = config['inputs']
    get_model_func = config['get_model_func']
    output_vars, cost_var = get_model_func(input_vars, is_training=False)

    describe_model()

    sess = tf.Session(config=sess_config)
    sess_init.init(sess)

    def run_input(dp):
        feed = dict(zip(input_vars, dp))
        results = sess.run(
            [cost_var] + output_vars, feed_dict=feed)
        cost = results[0]
        outputs = results[1:]
        return cost, outputs
    return run_input

class DatasetPredictor(object):
    def __init__(self, predict_config, dataset):
        assert isinstance(dataset, DataFlow)
        self.ds = dataset
        self.predict_func = get_predict_func(predict_config)

    def get_result(self):
        """ a generator to return prediction for each data"""
        for dp in self.ds.get_data():
            yield self.predict_func(dp)

    def get_all_result(self):
        return list(self.get_result())
