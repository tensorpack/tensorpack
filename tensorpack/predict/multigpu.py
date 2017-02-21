#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

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
                config.model.build_graph(config.model.get_reused_placehdrs())
            build_prediction_graph(fn, towers)

            self.sess = config.session_creator.create_session()
            config.session_init.init(self.sess)

            input_tensors = get_tensors_by_names(config.input_names)

            for k in towers:
                output_tensors = get_tensors_by_names(
                    [TowerContext.get_predict_towre_name('', k) + '/' + n
                     for n in config.output_names])
                self.predictors.append(OnlinePredictor(
                    input_tensors, output_tensors, config.return_input, self.sess))

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
        Its input is: [input[0] in tower[0], input[1] in tower[0], ...,
                      input[0] in tower[1], input[1] in tower[1], ...]
        And same for the output.
    """

    def __init__(self, config, towers):
        """
        Args:
            config (PredictConfig): the config to use.
            towers: a list of relative GPU id.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            input_names = []
            output_tensors = []

            def build_tower(k):
                towername = TowerContext.get_predict_tower_name(k)
                # inputs (placeholders) for this tower only
                input_tensors = config.model.build_placeholders(prefix=towername + '/')
                config.model.build_graph(input_tensors)

                input_names.extend([t.name for t in input_tensors])
                output_tensors.extend(get_tensors_by_names(
                    [towername + '/' + n
                     for n in config.output_names]))

            build_prediction_graph(build_tower, towers)

            input_tensors = get_tensors_by_names(input_names)

            sess = config.session_creator.create_session()
            config.session_init.init(sess)
            super(DataParallelOfflinePredictor, self).__init__(
                input_tensors, output_tensors, config.return_input, sess)
