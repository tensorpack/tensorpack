#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tower.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import re
from ..utils.naming import PREDICT_TOWER

__all__ = ['get_current_tower_context', 'TowerContext']

_CurrentTowerContext = None


class TowerContext(object):
    """ A context where the current model is being built in. """

    def __init__(self, tower_name, is_training=None):
        """
        Args:
            tower_name (str): 'tower0', 'towerp0', or ''
            is_training (bool): if None, automatically determine from tower_name.
        """
        self._name = tower_name
        if is_training is None:
            is_training = not self._name.startswith(PREDICT_TOWER)
        self._is_training = is_training

    @property
    def is_main_training_tower(self):
        return self.is_training and (self._name == '' or self._name == 'tower0')

    @property
    def is_main_tower(self):
        return self._name == '' or self._name == 'tower0'

    @property
    def is_training(self):
        return self._is_training

    @property
    def name(self):
        return self._name

    def get_variable_on_tower(self, *args, **kwargs):
        """
        Get a variable for this tower specifically, without reusing, even if
        it is called under a ``reuse=True`` variable scope.

        Tensorflow doesn't allow us to disable reuse under a
        ``reuse=True`` scope. This method provides a work around.
        See https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html#basics-of-tfvariable-scope

        Args:
            args: same as ``tf.get_variable()``.
        """
        with tf.variable_scope(self._name) as scope:
            with tf.variable_scope(scope, reuse=False):
                scope = tf.get_variable_scope()
                assert not scope.reuse
                return tf.get_variable(*args, **kwargs)

    def find_tensor_in_main_tower(self, graph, name):
        if self.is_main_tower:
            return graph.get_tensor_by_name(name)
        if name.startswith(PREDICT_TOWER):
            predict_tower_prefix = '{}[0-9]+/'.format(PREDICT_TOWER)
            newname = re.sub(predict_tower_prefix, '', name)
            try:
                return graph.get_tensor_by_name(newname)
            except KeyError:
                newname = re.sub(predict_tower_prefix, 'tower0/', name)
                return graph.get_tensor_by_name(newname)

    @staticmethod
    def get_predict_tower_name(towerid=0, prefix=''):
        """
        Args:
            towerid(int): an integer, the id of this predict tower, usually
                used to choose the GPU id.
            prefix(str): an alphanumeric prefix.
        Returns:
            str: the final tower name used to create a predict tower.
                Currently it is ``PREDICT_TOWER + prefix + towerid``.
        """
        assert prefix == '' or prefix.isalnum()
        return PREDICT_TOWER + prefix + str(towerid)

    def __enter__(self):
        global _CurrentTowerContext
        assert _CurrentTowerContext is None, \
            "Nesting TowerContext!"
        _CurrentTowerContext = self
        # TODO enter name_scope(None) first
        if len(self._name):
            self._scope = tf.name_scope(self._name)
            return self._scope.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CurrentTowerContext
        _CurrentTowerContext = None
        if len(self._name):
            self._scope.__exit__(exc_type, exc_val, exc_tb)
        return False

    def __str__(self):
        return "TowerContext(name={}, is_training={})".format(
            self._name, self._is_training)


def get_current_tower_context():
    global _CurrentTowerContext
    return _CurrentTowerContext
