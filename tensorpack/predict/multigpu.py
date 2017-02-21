#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from ..utils import logger
from ..tfutils import get_tensors_by_names, TowerContext, get_op_tensor_name
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
        assert len(towers) > 0
        self.graph = config._maybe_create_graph()
        self.predictors = []
        with self.graph.as_default():
            placeholder_names = set([k.name for k in config.model.get_inputs_desc()])

            def fn(_):
                config.model.build_graph(config.model.get_reused_placehdrs())
            build_prediction_graph(fn, towers)

            self.sess = config.session_creator.create_session()
            config.session_init.init(self.sess)

            get_tensor_fn = MultiTowerOfflinePredictor.get_tensors_maybe_in_tower
            for k in towers:
                input_tensors = get_tensor_fn(placeholder_names, config.input_names, k)
                output_tensors = get_tensor_fn(placeholder_names, config.output_names, k)
                self.predictors.append(OnlinePredictor(
                    input_tensors, output_tensors, config.return_input, self.sess))

    @staticmethod
    def get_tensors_maybe_in_tower(placeholder_names, names, k):
        def maybe_inside_tower(name):
            name = get_op_tensor_name(name)[0]
            if name in placeholder_names:
                return name
            else:
                # if the name is not a placeholder, use it's name in each tower
                return TowerContext.get_predict_tower_name(k) + '/' + name
        names = map(maybe_inside_tower, names)
        tensors = get_tensors_by_names(names)
        return tensors

    def _do_call(self, dp):
        # use the first tower for compatible PredictorBase interface
        return self.predictors[0]._do_call(dp)

    def get_predictor(self, n):
        """
        Returns:
            PredictorBase: the nth predictor on the nth tower.
        """
        l = len(self.predictors)
        if n >= l:
            logger.warn("n > #towers, will assign predictor to GPU by round-robin")
        return [self.predictors[k % l] for k in range(n)]

    def get_predictors(self):
        """
        Returns:
            list[PredictorBase]: a list of predictor
        """
        return self.predictors


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
        self.graph = config._maybe_create_graph()
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
