# -*- coding: UTF-8 -*-
# File: predict.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count
import argparse
from collections import namedtuple
import numpy as np
from tqdm import tqdm
from six.moves import zip

from .tfutils import *
from .utils import logger
from .tfutils.modelutils import describe_model
from .dataflow import DataFlow, BatchData

__all__ = ['PredictConfig', 'DatasetPredictor', 'get_predict_func']

class PredictConfig(object):
    def __init__(self, **kwargs):
        """
        The config used by `get_predict_func`.

        :param session_config: a `tf.ConfigProto` instance to instantiate the
            session. default to a session running 1 GPU.
        :param session_init: a `utils.sessinit.SessionInit` instance to
            initialize variables of a session.
        :param input_data_mapping: Decide the mapping from each component in data
            to the input tensor, since you may not need all input variables
            of the graph to run the graph for prediction (for example
            the `label` input is not used if you only need probability
            distribution).
            It should be a list with size=len(data_point),
            where each element is an index of the input variables each
            component of the data point should be fed into.
            If not given, defaults to range(len(input_vars))

            For example, in image classification task, the testing
            dataset only provides datapoints of images (no labels). When
            the input variables of the model is: ::

                input_vars: [image_var, label_var]

            the mapping should look like: ::

                input_data_mapping: [0] # the first component in a datapoint should map to `image_var`

        :param model: a `ModelDesc` instance
        :param output_var_names: a list of names of the output variables to predict, the
            variables can be any computable tensor in the graph.
            Predict specific output might not require all input variables.
        """
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__
        self.session_config = kwargs.pop('session_config', get_default_sess_config())
        assert_type(self.session_config, tf.ConfigProto)
        self.session_init = kwargs.pop('session_init')
        self.model = kwargs.pop('model')
        self.input_data_mapping = kwargs.pop('input_data_mapping', None)
        self.output_var_names = kwargs.pop('output_var_names')
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

def get_predict_func(config):
    """
    :param config: a `PredictConfig` instance.
    :returns: A prediction function that takes a list of input values, and return
        a list of output values defined in ``config.output_var_names``.
    """
    output_var_names = config.output_var_names

    # input/output variables
    input_vars = config.model.get_input_vars()
    cost_var = config.model.get_cost(input_vars, is_training=False)
    if config.input_data_mapping is None:
        input_map = input_vars
    else:
        input_map = [input_vars[k] for k in config.input_data_mapping]

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
        feed = dict(zip(input_map, dp))
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
    """
    Run the predict_config on a given `DataFlow`.
    """
    def __init__(self, predict_config, dataset, batch=0):
        """
        :param predict_config: a `PredictConfig` instance.
        :param dataset: a `DataFlow` instance.
        :param batch: if batch > zero, will batch the dataset before running.
        """
        assert isinstance(dataset, DataFlow)
        self.ds = dataset
        if batch > 0:
            self.ds = BatchData(self.ds, batch, remainder=True)
        self.predict_func = get_predict_func(predict_config)

    def get_result(self):
        """ A generator to produce prediction for each data"""
        with tqdm(total=self.ds.size()) as pbar:
            for dp in self.ds.get_data():
                yield PredictResult(dp, self.predict_func(dp))
                pbar.update()

    def get_all_result(self):
        """
        Run over the dataset and return a list of all predictions.
        """
        return list(self.get_result())
