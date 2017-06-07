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

    def __init__(self, tower_name,
                 device=None, is_training=None,
                 var_strategy='shared',
                 vs_name=None):
        """
        Args:
            tower_name (str): 'tower0', 'towerp0', or ''
            device (str or device function): the device to use. Defaults to either cpu0 or gpu0.
            is_training (bool): if None, automatically determine from tower_name.
            var_strategy (str): either 'shared' or 'replicated'.
            vs_name (str): the variable scope name to open. Only valid in
                'replicated' mode. Defaults to be tower_name.
        """
        self._name = tower_name
        self._device = device

        if is_training is None:
            is_training = not self._name.startswith(PREDICT_TOWER)
        self._is_training = is_training

        assert var_strategy in ['replicated', 'shared'], var_strategy
        self._var_strategy = var_strategy
        if self._var_strategy == 'replicated':
            assert self._name
            if vs_name is None:
                self._vs_name = self._name
            else:
                self._vs_name = vs_name
        else:
            assert vs_name is None, "vs_name is only valid in 'replicated' mode!"
            self._vs_name = ''

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
    def has_own_variables(self):
        return self._var_strategy == 'replicated'

    @property
    def name(self):
        return self._name

    # variable_scope name
    @property
    def vs_name(self):
        return self._vs_name

    @property
    def index(self):
        if self._name == '':
            return 0
        return int(self._name[-1])

    @property
    def device(self):
        return self._device

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
        self._ctxs = []
        if len(self._name):
            if self.has_own_variables:
                if len(self.vs_name):
                    self._ctxs.append(tf.variable_scope(self.vs_name))
            else:
                if self.is_training:
                    reuse = self.index > 0
                    if reuse is True:
                        self._ctxs.append(tf.name_scope(None))
                        self._ctxs.append(tf.variable_scope(
                            tf.get_variable_scope(), reuse=True))
                # if not training, should handle vs outside (TODO not good)
                self._ctxs.append(tf.name_scope(self._name))
        if self._device is not None:
            self._ctxs.append(tf.device(self._device))
        for c in self._ctxs:
            c.__enter__()

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
