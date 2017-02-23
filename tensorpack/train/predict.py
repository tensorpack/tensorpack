#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: predict.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from ..predict import (OnlinePredictor,
                       PredictorTowerBuilder)

__all__ = ['PredictorFactory']


class PredictorFactory(object):
    """ Make predictors from a trainer."""

    def __init__(self, trainer):
        """
        Args:
            towers (list[int]): list of gpu id
        """
        self.model = trainer.model
        self.towers = trainer.config.predict_tower

        def fn(_):
            self.model.build_graph(self.model.get_reused_placehdrs())
        self._tower_builder = PredictorTowerBuilder(fn)
        assert isinstance(self.towers, list)

    def get_predictor(self, input_names, output_names, tower):
        """
        Args:
            tower (int): need the kth tower (not the gpu id, but the id in TrainConfig.predict_tower)
        Returns:
            an online predictor (which has to be used under a default session)
        """
        tower = self.towers[tower]
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            # just ensure the tower exists. won't rebuild
            self._tower_builder.build(tower)

        placeholder_names = set([k.name for k in self.model.get_inputs_desc()])
        get_tensor_fn = PredictorTowerBuilder.get_tensors_maybe_in_tower
        in_tensors = get_tensor_fn(placeholder_names, input_names, tower)
        out_tensors = get_tensor_fn(placeholder_names, output_names, tower)
        return OnlinePredictor(in_tensors, out_tensors)
