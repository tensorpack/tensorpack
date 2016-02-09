#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: predict.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count, izip
import argparse
from collections import namedtuple
import numpy as np
from tqdm import tqdm

from utils import *
from utils.modelutils import describe_model
from utils import logger
from dataflow import DataFlow, BatchData

class PredictConfig(object):
    def __init__(self, **kwargs):
        """
        The config used by `get_predict_func`
        Args:
            session_config: a tf.ConfigProto instance to instantiate the
                session. default to a session running 1 GPU.
            session_init: a tensorpack.utils.sessinit.SessionInit instance to
                initialize variables of a session.
            input_data_mapping: Decide the mapping from each component in data
                to the input tensor, since you may not need all input variables
                of the graph to run the graph for prediction (for example
                the `label` input is not used if you only need probability
                distribution).
                It should be a list with size=len(one_data_point),
                where each element is an index of the input variables each
                component of the data point should be fed into.
                If not given, defaults to range(len(input_vars))

                For example, with image classification task, the testing
                dataset only provides datapoints of images (no labels). The
                arguments should look like:
                    inputs: [image_var, label_var]
                    input_data_mapping: [0]
                If this argument is not set, the inputs and the data points won't be aligned.
            model: a ModelDesc instance
            output_var_names: a list of names of the output variable to predict, the
                variables can be any computable tensor in the graph.
                if None, will only calculate the cost returned by `get_model_func`.
                Predict only specific output (instead of the cost)
                might be faster and might require only some of the input variables.
        """
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__
        self.session_config = kwargs.pop('session_config', get_default_sess_config())
        assert_type(self.session_config, tf.ConfigProto)
        self.session_init = kwargs.pop('session_init')
        self.model = kwargs.pop('model')
        self.input_data_mapping = kwargs.pop('input_dataset_mapping', None)
        self.output_var_names = kwargs.pop('output_var_names', None)
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

def get_predict_func(config):
    """
    Args:
        config: a PredictConfig
    Returns:
        A prediction function that takes a list of inputs value, and return
        one/a list of output values.
        If `output_var_names` is set, then the prediction function will
        return a list of output values. If not, will return a cost.
    """
    output_var_names = config.output_var_names

    # input/output variables
    input_vars = config.model.get_input_vars()
    cost_var = config.model.get_cost(input_vars, is_training=False)
    input_map = [input_vars[k] for k in config.input_data_mapping]
    if input_map is None:
        input_map = input_vars

    # check output_var_names against output_vars
    if output_var_names is not None:
        output_vars = [tf.get_default_graph().get_tensor_by_name(n) for n in output_var_names]
    else:
        output_vars = []

    describe_model()

    sess = tf.Session(config=config.session_config)
    config.session_init.init(sess)

    def run_input(dp):
        assert len(input_map) == len(dp), \
            "Graph has {} inputs but dataset only gives {} components!".format(
                    len(input_map), len(dp))
        feed = dict(izip(input_map, dp))
        if output_var_names is not None:
            results = sess.run(output_vars, feed_dict=feed)
            return results
        else:
            results = sess.run([cost_var], feed_dict=feed)
            cost = results[0]
            return cost
    return run_input

PredictResult = namedtuple('PredictResult', ['input', 'output'])

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
        with tqdm(total=self.ds.size()) as pbar:
            for dp in self.ds.get_data():
                yield PredictResult(dp, self.predict_func(dp))
                pbar.update()

    def get_all_result(self):
        return list(self.get_result())
