#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tower.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from .common import get_tf_version_number

__all__ = ['get_current_tower_context', 'TowerContext']

_CurrentTowerContext = None


class TowerContext(object):
    """ A context where the current model is being built in. """

    def __init__(self, tower_name, is_training, index=0, use_vs=False):
        """
        Args:
            tower_name (str): The name scope of the tower.
            is_training (bool):
            index (int): index of this tower, only used in training.
            use_vs (bool): Open a new variable scope with this name.
        """
        self._name = tower_name
        self._is_training = bool(is_training)

        if not self._is_training:
            assert index == 0 and not use_vs, \
                "use_vs and index are only used in training!"

        self._index = int(index)
        if use_vs:
            self._vs_name = self._name
            assert len(self._name)
        else:
            self._vs_name = ''

        if self.has_own_variables:
            assert not tf.get_variable_scope().reuse, "reuse=True in tower {}!".format(tower_name)

    @property
    def is_main_training_tower(self):
        return self.is_training and self._index == 0

    @property
    def is_training(self):
        return self._is_training

    @property
    def has_own_variables(self):
        """
        Whether this tower is supposed to have its own variables.
        """
        return self.is_main_training_tower or len(self._vs_name) > 0

    @property
    def name(self):
        return self._name

    @property
    def vs_name(self):
        return self._vs_name

    def filter_vars_by_vs_name(self, varlist):
        """
        Filter the list and only keep those under the current variable scope.
        If this tower doesn't contain its own variable scope, return the list as-is.

        Args:
            varlist (list[tf.Variable] or list[tf.Tensor]):
        """
        if not self.has_own_variables:
            return varlist
        if len(self._vs_name) == 0:
            # main_training_tower with no name. assume no other towers has
            # been built yet, then varlist contains vars only in the first tower.
            return varlist
        prefix = self._vs_name + '/'
        return [v for v in varlist if v.op.name.startswith(prefix)]

    @property
    def index(self):
        return self._index

    def __enter__(self):
        global _CurrentTowerContext
        assert _CurrentTowerContext is None, \
            "Nesting TowerContext!"
        _CurrentTowerContext = self
        self._ctxs = []
        curr_vs = tf.get_variable_scope()
        assert curr_vs.name == '', "Nesting TowerContext with an existing variable scope!"
        # assert empty name scope as well (>1.2.1?)
        if len(self._name):
            if not self.is_training:
                # if not training, should handle reuse outside
                # but still good to clear name_scope first
                self._ctxs.append(tf.name_scope(None))
                self._ctxs.append(tf.name_scope(self._name))
            else:
                if self.has_own_variables:
                    if len(self._vs_name):
                        self._ctxs.append(tf.variable_scope(self._vs_name))
                    else:
                        self._ctxs.append(tf.name_scope(self._name))
                else:
                    reuse = self._index > 0
                    if reuse:
                        self._ctxs.append(tf.variable_scope(
                            tf.get_variable_scope(), reuse=True))
                    self._ctxs.append(tf.name_scope(self._name))
        for c in self._ctxs:
            c.__enter__()

        if get_tf_version_number() >= 1.2:
            ns = tf.get_default_graph().get_name_scope()
            assert ns == self._name, \
                "Name conflict: name_scope inside tower '{}' becomes '{}'!".format(self._name, ns) \
                + " You may need a different name for the tower!"

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CurrentTowerContext
        _CurrentTowerContext = None
        for c in self._ctxs[::-1]:
            c.__exit__(exc_type, exc_val, exc_tb)
        return False

    def __str__(self):
        return "TowerContext(name={}, is_training={})".format(
            self._name, self._is_training)


def get_current_tower_context():
    global _CurrentTowerContext
    return _CurrentTowerContext
