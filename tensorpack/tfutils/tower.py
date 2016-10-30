#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tower.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import re

__all__ = ['get_current_tower_context', 'TowerContext']

_CurrentTowerContext = None

class TowerContext(object):
    def __init__(self, tower_name, is_training=None):
        """ tower_name: 'tower0', 'towerp0', or '' """
        self._name = tower_name
        if is_training is None:
            is_training = not self._name.startswith('towerp')
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
        Get a variable for this tower specifically, without reusing.
        Tensorflow doesn't allow reuse=False scope under a
        reuse=True scope. This method provides a work around.
        See https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html#basics-of-tfvariable-scope

        :param args, kwargs: same as tf.get_variable()
        """
        with tf.variable_scope(self._name) as scope:
            with tf.variable_scope(scope, reuse=False):
                scope = tf.get_variable_scope()
                assert scope.reuse == False
                return tf.get_variable(*args, **kwargs)

    def find_tensor_in_main_tower(self, graph, name):
        if self.is_main_tower:
            return graph.get_tensor_by_name(name)
        if name.startswith('towerp'):
            newname = re.sub('towerp[0-9]+/', '', name)
            try:
                return graph.get_tensor_by_name(newname)
            except KeyError:
                newname = re.sub('towerp[0-9]+/', 'tower0/', name)
                return graph.get_tensor_by_name(newname)

    def __enter__(self):
        global _CurrentTowerContext
        assert _CurrentTowerContext is None, \
                "Nesting TowerContext!"
        _CurrentTowerContext = self
        if len(self._name):
            self._scope = tf.name_scope(self._name)
            return self._scope.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CurrentTowerContext
        _CurrentTowerContext = None
        if len(self._name):
            self._scope.__exit__(exc_type, exc_val, exc_tb)
        return False

def get_current_tower_context():
    global _CurrentTowerContext
    return _CurrentTowerContext

