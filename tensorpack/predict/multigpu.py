#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..utils import logger
from ..utils.naming import PREDICT_TOWER
from ..tfutils import get_tensors_by_names, TowerContext
from .base import OnlinePredictor, build_prediction_graph

__all__ = ['MultiTowerOfflinePredictor',
           'DataParallelOfflinePredictor']


class MultiTowerOfflinePredictor(OnlinePredictor):
    """ A multi-tower multi-GPU predictor. """

    def __init__(self, config, towers):
        """
        Args:
            config (PredictConfig): the config to use.
            towers: a list of relative GPU id.
        """
        self.graph = tf.Graph()
        self.predictors = []
        with self.graph.as_default():
            # TODO backup summary keys?
            def fn(_):
                config.model.build_graph(config.model.get_input_vars())
            build_prediction_graph(fn, towers)

            self.sess = tf.Session(config=config.session_config)
            config.session_init.init(self.sess)

            input_vars = get_tensors_by_names(config.input_names)

            for k in towers:
                output_vars = get_tensors_by_names(
                    ['{}{}/'.format(PREDICT_TOWER, k) + n
                     for n in config.output_names])
                self.predictors.append(OnlinePredictor(
                    self.sess, input_vars, output_vars, config.return_input))

    def _do_call(self, dp):
        # use the first tower for compatible PredictorBase interface
        return self.predictors[0]._do_call(dp)

    def get_predictors(self, n):
        """
        Returns:
            PredictorBase: the nth predictor on the nth GPU.
        """
        return [self.predictors[k % len(self.predictors)] for k in range(n)]


class DataParallelOfflinePredictor(OnlinePredictor):
    """ A data-parallel predictor.
    It runs different towers in parallel.
    """

    def __init__(self, config, towers):
        """
        Args:
            config (PredictConfig): the config to use.
            towers: a list of relative GPU id.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            sess = tf.Session(config=config.session_config)
            input_var_names = []
            output_vars = []
            for idx, k in enumerate(towers):
                towername = PREDICT_TOWER + str(k)
                input_vars = config.model.build_placeholders(
                    prefix=towername + '-')
                logger.info(
                    "Building graph for predictor tower {}...".format(k))
                with tf.device('/gpu:{}'.format(k) if k >= 0 else '/cpu:0'), \
                        TowerContext(towername, is_training=False), \
                        tf.variable_scope(tf.get_variable_scope(),
                                          reuse=True if idx > 0 else None):
                    config.model.build_graph(input_vars)
                input_var_names.extend([k.name for k in input_vars])
                output_vars.extend(get_tensors_by_names(
                    [towername + '/' + n
                     for n in config.output_names]))

            input_vars = get_tensors_by_names(input_var_names)
            config.session_init.init(sess)
            super(DataParallelOfflinePredictor, self).__init__(
                sess, input_vars, output_vars, config.return_input)
