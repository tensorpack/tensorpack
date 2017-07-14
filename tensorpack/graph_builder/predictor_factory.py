#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: predictor_factory.py

import tensorflow as tf
# from ..tfutils.tower import TowerContext
from ..predict import (OnlinePredictor,
                       PredictorTowerBuilder)

__all__ = ['PredictorFactory']


# class PredictorTowerBuilder(object):
#     def __init__(self, model):
#         self._model = model
#         self._towers = []
#
#     def build(self, tower_name, device, input=None):
#         with tf.device(device), TowerContext(tower_name, is_training=False):
#             if input is None:
#                 input = self._model.get_reused_placehdrs()
#             self._model.build_graph(input)
#
#

# SMART
class PredictorFactory(object):
    """ Make predictors from :class:`ModelDesc` and cache them."""

    def __init__(self, model, towers, vs_name):
        """
        Args:
            towers (list[int]): list of available gpu id
        """
        self.model = model
        self.towers = towers
        self.vs_name = vs_name

        def fn(_):
            self.model.build_graph(self.model.get_reused_placehdrs())
        self._tower_builder = PredictorTowerBuilder(fn)
        assert isinstance(self.towers, list), self.towers

    def get_predictor(self, input_names, output_names, tower):
        """
        Args:
            tower (int): need the kth tower (not the gpu id, but the id in TrainConfig.predict_tower)
        Returns:
            an online predictor (which has to be used under a default session)
        """
        tower = self.towers[tower]
        # just ensure the tower exists. won't rebuild (memoized)
        with tf.variable_scope(self.vs_name, reuse=True):
            self._tower_builder.build(tower)

        placeholder_names = set([k.name for k in self.model.get_inputs_desc()])
        get_tensor_fn = PredictorTowerBuilder.get_tensors_maybe_in_tower
        in_tensors = get_tensor_fn(placeholder_names, input_names, tower)
        out_tensors = get_tensor_fn(placeholder_names, output_names, tower)
        return OnlinePredictor(in_tensors, out_tensors)
