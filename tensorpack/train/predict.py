#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: predict.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from ..utils import SUMMARY_BACKUP_KEYS, PREDICT_TOWER
from ..tfutils.collection import freeze_collection
from ..utils.argtools import memoized
from ..tfutils import get_tensors_by_names, get_op_tensor_name
from ..predict import OnlinePredictor, build_prediction_graph

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
        assert isinstance(self.towers, list)

    # TODO sess option
    def get_predictor(self, input_names, output_names, tower):
        """
        Args:
            tower (int): need the kth tower (not the gpu id)
        Returns:
            an online predictor (which has to be used under a default session)
        """
        self._build_predict_tower()
        tower = self.towers[tower]

        placeholder_names = set([k.name for k in self.model.get_inputs_desc()])

        def get_name_in_tower(name):
            return PREDICT_TOWER + str(tower) + '/' + name

        def maybe_inside_tower(name):
            name = get_op_tensor_name(name)[0]
            if name in placeholder_names:
                return name
            else:
                return get_name_in_tower(name)

        input_names = map(maybe_inside_tower, input_names)
        raw_input_tensors = get_tensors_by_names(input_names)

        output_names = map(get_name_in_tower, output_names)
        output_tensors = get_tensors_by_names(output_names)
        return OnlinePredictor(raw_input_tensors, output_tensors)

    @memoized
    def _build_predict_tower(self):
        # build_predict_tower might get called anywhere, but 'PREDICT_TOWER'
        # should always be the outermost name scope
        with tf.name_scope(None), \
                freeze_collection(SUMMARY_BACKUP_KEYS), \
                tf.variable_scope(tf.get_variable_scope(), reuse=True):
            def fn(_):
                self.model.build_graph(self.model.get_reused_placehdrs())
            build_prediction_graph(fn, self.towers)
