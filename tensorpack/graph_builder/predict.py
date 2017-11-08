#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: predict.py

import tensorflow as tf
from contextlib import contextmanager

from ..utils import logger
from ..tfutils.tower import TowerContext
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
                    self._ns_name, is_training=False, vs_name=self._vs_name):
            inputs = input.get_input_tensors()
            assert isinstance(inputs, (list, tuple)), inputs
            return tower_fn(*inputs)
