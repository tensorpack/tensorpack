#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tower.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from six.moves import zip

from ..utils import logger
from .common import get_tf_version_number, get_op_or_tensor_by_name, get_op_tensor_name

__all__ = ['get_current_tower_context', 'TowerContext', 'TowerFuncWrapper']

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

    # TODO clarify the interface on name/vs_name/ns_name.
    # TODO in inference, vs_name may need to be different from ns_name.i
    # How to deal with this?
    @property
    def name(self):
        return self._name

    @property
    def vs_name(self):
        return self._vs_name

    @property
    def ns_name(self):
        return self._name

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
        assert _CurrentTowerContext is None, "Cannot nest TowerContext!"
        _CurrentTowerContext = self
        self._ctxs = []
        curr_vs = tf.get_variable_scope()
        assert curr_vs.name == '', "Cannot nest TowerContext with an existing variable scope!"

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
            # check that ns_name is always the same as _name
            ns = tf.get_default_graph().get_name_scope()
            assert ns == self._name, \
                "Name conflict: name_scope inside tower '{}' becomes '{}'!".format(self._name, ns) \
                + " You may need a different name for the tower!"
        return self

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


class TowerFuncWrapper(object):
    """
    A wrapper around a function which builds one tower (one replicate of the model).
    It keeps track of the name scope, variable scope and input/output tensors
    each time the function is called.
    """

    def __init__(self, tower_fn, inputs_desc=None):
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

            self._towers = []

    def __new__(cls, tower_fn, inputs_desc=None):
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
        self._towers.append(handle)
        return output

    @property
    def towers(self):
        # TODO another wrapper around towerhandlelist
        return self._towers


class TowerTensorHandle(object):
    """
    When a function is called multiple times under each tower,
    it becomes hard to keep track of the scope and access those tensors
    in each tower.
    This class provides easy access to the tensors as well as the
    inputs/outputs created in each tower.
    """

    # TODO hide it from doc
    def __init__(self, ctx, input, output, inputs_desc=None):
        """
        Don't use it because you never need to create the handle by yourself.
        """
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

    # def make_callable(self, input_names, output_names):
    #     input_tensors = self.get_tensors(input_names)
    #     output_tensors = self.get_tensors(output_names)
    #     pass
