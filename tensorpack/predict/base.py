#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta, abstractproperty
import tensorflow as tf
import six

from ..utils import logger
from ..tfutils import get_tensors_by_names, TowerContext

__all__ = ['OnlinePredictor', 'OfflinePredictor',
        'AsyncPredictorBase',
        'MultiTowerOfflinePredictor', 'build_multi_tower_prediction_graph']


class PredictorBase(object):
    __metaclass__ = ABCMeta
    """
    Property:
    session
    return_input
    """

    def __call__(self, *args):
        """
        if len(args) == 1, assume args[0] is a datapoint (a list)
        else, assume args is a datapoinnt
        """
        if len(args) != 1:
            dp = args
        else:
            dp = args[0]
        output = self._do_call(dp)
        if self.return_input:
            return (dp, output)
        else:
            return output

    @abstractmethod
    def _do_call(self, dp):
        """
        :param dp: input datapoint.  must have the same length as input_names
        :return: output as defined by the config
        """

class AsyncPredictorBase(PredictorBase):
    @abstractmethod
    def put_task(self, dp, callback=None):
        """
        :param dp: A data point (list of component) as inputs.
            (It should be either batched or not batched depending on the predictor implementation)
        :param callback: a thread-safe callback to get called with
            either outputs or (inputs, outputs)
        :return: a Future of results
        """

    @abstractmethod
    def start(self):
        """ Start workers """

    def _do_call(self, dp):
        assert six.PY3, "With Python2, sync methods not available for async predictor"
        fut = self.put_task(dp)
        # in Tornado, Future.result() doesn't wait
        return fut.result()

class OnlinePredictor(PredictorBase):
    def __init__(self, sess, input_tensors, output_tensors, return_input=False):
        self.session = sess
        self.return_input = return_input

        self.input_tensors = input_tensors
        self.output_tensors = output_tensors

    def _do_call(self, dp):
        assert len(dp) == len(self.input_tensors), \
            "{} != {}".format(len(dp), len(self.input_tensors))
        feed = dict(zip(self.input_tensors, dp))
        output = self.session.run(self.output_tensors, feed_dict=feed)
        return output


class OfflinePredictor(OnlinePredictor):
    """ Build a predictor from a given config, in an independent graph"""
    def __init__(self, config):
        self.graph = tf.Graph()
        with self.graph.as_default():
            input_vars = config.model.get_input_vars()
            with TowerContext('', False):
                config.model.build_graph(input_vars)

            input_vars = get_tensors_by_names(config.input_names)
            output_vars = get_tensors_by_names(config.output_names)

            sess = tf.Session(config=config.session_config)
            config.session_init.init(sess)
            super(OfflinePredictor, self).__init__(
                    sess, input_vars, output_vars, config.return_input)


def build_multi_tower_prediction_graph(model, towers):
    """
    :param towers: a list of gpu relative id.
    """
    input_vars = model.get_input_vars()
    for k in towers:
        logger.info(
"Building graph for predictor tower {}...".format(k))
        with tf.device('/gpu:{}'.format(k) if k >= 0 else '/cpu:0'), \
                TowerContext('towerp{}'.format(k)):
            model.build_graph(input_vars)
            tf.get_variable_scope().reuse_variables()

class MultiTowerOfflinePredictor(OnlinePredictor):
    def __init__(self, config, towers):
        self.graph = tf.Graph()
        self.predictors = []
        with self.graph.as_default():
            # TODO backup summary keys?
            build_multi_tower_prediction_graph(config.model, towers)

            self.sess = tf.Session(config=config.session_config)
            config.session_init.init(self.sess)

            input_vars = get_tensors_by_names(config.input_names)

            for k in towers:
                output_vars = get_tensors_by_names(
                        ['{}{}/'.format(self.PREFIX, k) + n \
                                for n in config.output_names])
                self.predictors.append(OnlinePredictor(
                    self.sess, input_vars, output_vars, config.return_input))

    def _do_call(self, dp):
        # use the first tower for compatible PredictorBase interface
        return self.predictors[0]._do_call(dp)

    def get_predictors(self, n):
        return [self.predictors[k % len(self.predictors)] for k in range(n)]
