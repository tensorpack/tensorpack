#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: predict.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count
import argparse
import numpy as np

from utils import *
from utils.modelutils import describe_model
from utils import logger
from dataflow import DataFlow, BatchData

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

    # Provide this if only specific output is needed.
    # by default will evaluate all outputs as well as cost
    output_var_name = config.get('output_var', None)

    # input/output variables
    input_vars = config['inputs']
    get_model_func = config['get_model_func']
    output_vars, cost_var = get_model_func(input_vars, is_training=False)

    describe_model()

    sess = tf.Session(config=sess_config)
    sess_init.init(sess)

    def run_input(dp):
        # TODO if input and dp not aligned?
        feed = dict(zip(input_vars, dp))
        if output_var_name is not None:
            fetches = tf.get_default_graph().get_tensor_by_name(output_var_name)
            results = sess.run(fetches, feed_dict=feed)
            return results[0]
        else:
            fetches = [cost_var] + output_vars
            results = sess.run(fetches, feed_dict=feed)
            cost = results[0]
            outputs = results[1:]
            return cost, outputs
    return run_input

class DatasetPredictor(object):
    def __init__(self, predict_config, dataset, batch=0):
        """
        A predictor with the given predict_config, run on the given dataset
        if batch is larger than zero, the dataset will be batched
        """
        assert isinstance(dataset, DataFlow)
        self.ds = dataset
        if batch > 0:
            self.ds = BatchData(self.ds, batch, remainder=True)
        self.predict_func = get_predict_func(predict_config)

    def get_result(self):
        """ a generator to return prediction for each data"""
        for dp in self.ds.get_data():
            yield self.predict_func(dp)

    def get_all_result(self):
        return list(self.get_result())
