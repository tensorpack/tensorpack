#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tower.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from six.moves import zip

from ..utils import logger
from ..utils.argtools import call_only_once
from ..utils.naming import TRAIN_TOWER_FREEZE_KEYS, PREDICT_TOWER_FREEZE_KEYS
from ..utils.develop import HIDE_DOC
from .collection import CollectionGuard
from .common import get_tf_version_number, get_op_or_tensor_by_name, get_op_tensor_name

__all__ = ['get_current_tower_context', 'TowerContext', 'TowerFuncWrapper',
           'TowerTensorHandle', 'TowerTensorHandles']

_CurrentTowerContext = None


class TowerContext(object):
    """ A context where the current model is being built in. """

    def __init__(self, tower_name, is_training, index=0, vs_name=''):
        """
        Args:
            tower_name (str): The name scope of the tower.
            is_training (bool):
            index (int): index of this tower, only used in training.
            vs_name (str): Open a new variable scope with this name.
        """
        self._name = tower_name
        self._is_training = bool(is_training)

        if not self._is_training:
            assert index == 0, \
                "TowerContext(index) is only valid in training!"

        self._index = int(index)
        self._vs_name = vs_name
        if len(vs_name):
            assert len(tower_name), "TowerContext(vs_name) cannot be used with an empty tower_name!"

        self._initial_vs_reuse = tf.get_variable_scope().reuse
        if self.has_own_variables:
            assert not self._initial_vs_reuse, \
                "Cannot create tower {} with reuse=True!".format(tower_name)

        self._collection_guard = CollectionGuard(
            self._name,
            check_diff=not self.is_main_training_tower,
            freeze_keys=self._keys_to_freeze())

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
        return self.is_main_training_tower or \
            (self.is_training and len(self._vs_name) > 0) or \
            (not self.is_training and not self._initial_vs_reuse)

    @property
    def name(self):
        return self._name

    @property
    def vs_name(self):
        return self._vs_name

    @property
    def ns_name(self):
        return self._name

    def get_collection_in_tower(self, key):
        """
        Get items from this collection that are added in the current tower.
        """
        return self._collection_guard.get_collection_in_tower(key)

    @property
    def index(self):
        return self._index

    @call_only_once
    def _get_scopes(self):
        if not len(self._name):
            return []
        ret = []

        # either the Tower was originally created with reuse,
        # or a training tower without vs has to use reuse.
        reuse = (self.is_training and self._index > 0 and not
                 self.has_own_variables) or self._initial_vs_reuse

        if len(self._vs_name):
            ret.append(tf.variable_scope(self._vs_name, reuse=reuse))
        else:
            if reuse:
                ret.append(tf.variable_scope(
                    tf.get_variable_scope(), reuse=True))
        # always clear existing ns  # TODO check existing ns
        if len(self._name) and self._name != self._vs_name:
            ret.append(tf.name_scope(self._name + '/'))
        return ret

    def _keys_to_freeze(self):
        if self.is_main_training_tower:
            return []
        if self.is_training:
            return TRAIN_TOWER_FREEZE_KEYS
        return PREDICT_TOWER_FREEZE_KEYS

    def __enter__(self):
        global _CurrentTowerContext
        assert _CurrentTowerContext is None, "Cannot nest TowerContext!"
        _CurrentTowerContext = self
        curr_vs = tf.get_variable_scope()
        assert curr_vs.name == '', "Cannot nest TowerContext with an existing variable scope!"

        self._ctxs = self._get_scopes()
        self._ctxs.append(self._collection_guard)
        for c in self._ctxs:
            c.__enter__()

        if get_tf_version_number() >= 1.2:
            # check that ns_name is always the same as _name
            ns = tf.get_default_graph().get_name_scope()
            assert ns == self._name, \
                "Name conflict: name_scope inside tower '{}' becomes '{}'!".format(self._name, ns) \
                + " You may need a different name for the tower!"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CurrentTowerContext
        _CurrentTowerContext = None

        if not self.has_own_variables:
            diff_trainable_vars = self._collection_guard.get_collection_in_tower(tf.GraphKeys.TRAINABLE_VARIABLES)
            assert len(diff_trainable_vars) == 0,  \
                "New TRAINABLE_VARIABLES shouldn't be created in {}: ".format(
                    self._name) + ', '.join([k.name for k in diff_trainable_vars])
        for c in self._ctxs[::-1]:
            c.__exit__(exc_type, exc_val, exc_tb)
        return False

    def __str__(self):
        return "TowerContext(name={}, is_training={})".format(
            self._name, self._is_training)


def get_current_tower_context():
    return _CurrentTowerContext


class TowerFuncWrapper(object):
    """
    A wrapper around a function which builds one tower (one replicate of the model).
    It keeps track of the name scope, variable scope and input/output tensors
    each time the function is called.

    :class:`TowerTrainer` needs this option to be set, so that
    it knows how to build a predictor.
    """

    def __init__(self, tower_fn, inputs_desc):
        """
        Args:
            tower_func: a function which builds one tower in the graph.
                It takes several input tensors and could return anything.
            inputs_desc ([InputDesc]): use this to figure out the right name for the input tensors.
        """
        assert callable(tower_fn), tower_fn
        if not isinstance(tower_fn, TowerFuncWrapper):
            self._tower_fn = tower_fn
            self._inputs_desc = inputs_desc

            self._handles = []

    def __new__(cls, tower_fn, inputs_desc):
        # to avoid double-wrapping a function
        if isinstance(tower_fn, TowerFuncWrapper):
            return tower_fn
        else:
            return super(TowerFuncWrapper, cls).__new__(cls)

    def __call__(self, *args):
        ctx = get_current_tower_context()
        assert ctx is not None, "Function must be called under TowerContext!"
        output = self._tower_fn(*args)
        handle = TowerTensorHandle(ctx, args, output, self._inputs_desc)
        self._handles.append(handle)
        return output

    @property
    def towers(self):
        """
        Returns:
            a :class:`TowerTensorHandles` object, that can
            access the tower handles by either indices or names.
        """
        return TowerTensorHandles(self._handles)

    @property
    def inputs_desc(self):
        return self._inputs_desc


class TowerTensorHandles(object):
    """
    Wrap a list of :class:`TowerTensorHandle`,
    to support access to them by index or names.
    """
    def __init__(self, handles):
        self._handles = handles
        self._name_to_handle = {k.ns_name: k for k in handles}

    def __getitem__(self, name_or_index):
        """
        Args:
            name_or_index (str or int):

        Returns:
            a :class:`TowerTensorHandle`.
        """
        if isinstance(name_or_index, int):
            return self._handles[name_or_index]
        return self._name_to_handle[name_or_index]

    def training(self):
        """
        Returns:
            A :class:`TowerTensorHandles`, containing only the training towers.
        """
        handles = [h for h in self._handles if h.is_training]
        return TowerTensorHandles(handles)

    def inference(self):
        """
        Returns:
            A :class:`TowerTensorHandles`, containing only the inference towers.
        """
        handles = [h for h in self._handles if not h.is_training]
        return TowerTensorHandles(handles)


class TowerTensorHandle(object):
    """
    When a function is called multiple times under each tower,
    it becomes hard to keep track of the scope and access those tensors
    in each tower.
    This class provides easy access to the tensors as well as the
    inputs/outputs created in each tower.
    """

    @HIDE_DOC
    def __init__(self, ctx, input, output, inputs_desc=None):
        self._ctx = ctx

        self._extra_tensor_names = {}
        if inputs_desc is not None:
            assert len(inputs_desc) == len(input)
            self._extra_tensor_names = {
                get_op_tensor_name(x.name)[1]: y for x, y in zip(inputs_desc, input)}
        self._input = input
        self._output = output

    @property
    def vs_name(self):
        return self._ctx.vs_name

    @property
    def ns_name(self):
        return self._ctx.ns_name

    def get_tensor(self, name):
        """
        Get a tensor in this tower. The name can be:
        1. The name of the tensor without any tower prefix.
        2. The name of an :class:`InputDesc`, if it is used when building the tower.
        """
        name = get_op_tensor_name(name)[1]
        if len(self.ns_name):
            name_with_ns = self.ns_name + "/" + name
        else:
            name_with_ns = name

        try:
            ret = get_op_or_tensor_by_name(name_with_ns)
        except KeyError:
            if name in self._extra_tensor_names:
                return self._extra_tensor_names[name]
            raise
        else:
            if name in self._extra_tensor_names:
                logger.warn(
                    "'{}' may refer to both the tensor '{}' or the input '{}'.".format(
                        name, ret.name, self._extra_tensor_names[name].name) +
                    "Assuming it is the tensor '{}'.".format(ret.name))
            return ret

    def get_tensors(self, names):
        return [self.get_tensor(name) for name in names]

    def __getitem__(self, name):
        return self.get_tensor(name)

    def get_variable(self, name):
        """
        Get a variable used in this tower.
        """
        name = get_op_tensor_name(name)[1]
        if len(self.vs_name):
            name_with_vs = self.vs_name + "/" + name
        else:
            name_with_vs = name
        return get_op_or_tensor_by_name(name_with_vs)

    @property
    def input(self):
        """
        The list of input tensors used to build the tower.
        """
        return self._input

    @property
    def output(self):
        """
        The output returned by the tower function.
        """
        return self._output

    @property
    def is_training(self):
        return self._ctx.is_training
