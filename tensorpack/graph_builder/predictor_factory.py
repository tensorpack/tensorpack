#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: predictor_factory.py

import tensorflow as tf
from contextlib import contextmanager

from ..utils import logger
from ..tfutils.tower import TowerContext, TowerFuncWrapper
from ..tfutils.collection import freeze_collection
from ..utils.naming import TOWER_FREEZE_KEYS
from ..input_source import PlaceholderInput
from .training import GraphBuilder

__all__ = ['SimplePredictBuilder']


class SimplePredictBuilder(GraphBuilder):
    """
    Single-tower predictor.
    """
    def __init__(self, ns_name='', vs_name='', device=0):
        """
        Args:
            ns_name (str):
            vs_name (str):
            device (int):
        """
        self._ns_name = ns_name
        self._vs_name = vs_name

        device = '/gpu:{}'.format(device) if device >= 0 else '/cpu:0'
        self._device = device

    @contextmanager
    def _maybe_open_vs(self):
        if len(self._vs_name):
            with tf.variable_scope(self._vs_name):
                yield
        else:
            yield

    def build(self, input, tower_fn):
        """
        Args:
            input (InputSource): must have been setup
            tower_fn ( [tf.Tensors] ->): callable that takes input tensors.

        Returns:
            The return value of tower_fn called under the proper context.
        """
        assert input.setup_done()
        logger.info("Building predictor tower '{}' on device {} ...".format(
            self._ns_name, self._device))

        with tf.device(self._device), \
                self._maybe_open_vs(), \
                TowerContext(
                    self._ns_name, is_training=False, vs_name=self._vs_name), \
                freeze_collection(TOWER_FREEZE_KEYS + [tf.GraphKeys.UPDATE_OPS]):
                # also freeze UPDATE_OPS in inference, because they should never be used
                # TODO a better way to log and warn about collection change during build_graph.

            inputs = input.get_input_tensors()
            assert isinstance(inputs, (list, tuple)), inputs
            return tower_fn(*inputs)


class PredictorFactory(object):
    """ Make predictors from :class:`ModelDesc`."""

    def __init__(self, model, vs_name=''):
        """
        Args:
            model (ModelDesc):
            vs_name (str):
        """
        self._model = model
        self._vs_name = vs_name

        self._names_built = {}

    def build(self, tower_name, device, input=None):
        """
        Args:
            tower_name (str):
            device(str):
            input (InputSource): must be setup already. If None, will use InputDesc from the model.
        """
        logger.info("Building predictor tower '{}' on device {} ...".format(tower_name, device))
        assert tower_name not in self._names_built, \
            "Prediction tower with name '{}' already exists!".format(tower_name)

        with tf.device(device), \
                TowerContext(tower_name, is_training=False), \
                freeze_collection(TOWER_FREEZE_KEYS + [tf.GraphKeys.UPDATE_OPS]):
                # also freeze UPDATE_OPS in inference, because they should never be used
                # TODO a better way to log and warn about collection change during build_graph.
            inputs_desc = self._model.get_inputs_desc()
            if input is None:
                input = PlaceholderInput()
                input.setup(inputs_desc)
            inputs = input.get_input_tensors()
            assert isinstance(inputs, (list, tuple)), inputs

            def tower_func(*inputs):
                self._model.build_graph(inputs)

            tower_func = TowerFuncWrapper(tower_func, inputs_desc)
            tower_func(*inputs)

        self._names_built[tower_name] = tower_func.towers[0]
        return self._names_built[tower_name]

    def has_built(self, tower_name):
        return tower_name in self._names_built

    def get_predictor(self, input_names, output_names, tower):
        """
        Args:
            tower (int): use device '/gpu:{tower}' or use -1 for '/cpu:0'.
        Returns:
            an online predictor (which has to be used under a default session)
        """
        tower_name = 'towerp{}'.format(tower)
        device = '/gpu:{}'.format(tower) if tower >= 0 else '/cpu:0'
        # use a previously-built tower
        # TODO check conflict with inference runner??
        if tower_name not in self._names_built:
            with tf.variable_scope(self._vs_name, reuse=True):
                handle = self.build(tower_name, device)
        else:
            handle = self._names_built[tower_name]

        in_tensors = handle.get_tensors(input_names)
        out_tensors = handle.get_tensors(output_names)
        from ..predict import OnlinePredictor   # noqa TODO
        return OnlinePredictor(in_tensors, out_tensors)
