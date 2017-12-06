#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
from collections import namedtuple
import tensorflow as tf
import six

from ..utils.argtools import memoized
from ..utils.develop import log_deprecated
from ..tfutils.gradproc import FilterNoneGrad
from ..tfutils.tower import get_current_tower_context
from ..input_source import InputSource
from ..models.regularize import regularize_cost_from_collection

__all__ = ['InputDesc', 'ModelDesc', 'ModelDescBase']


class InputDesc(
        namedtuple('InputDescTuple', ['type', 'shape', 'name'])):
    """
    Metadata about an input entry point to the graph.
    This metadata can be later used to build placeholders or other types of
    input source.
    """

    def __new__(cls, type, shape, name):
        """
        Args:
            type (tf.DType):
            shape (tuple):
            name (str):
        """
        shape = tuple(shape)    # has to be tuple for "self" to be hashable
        self = super(InputDesc, cls).__new__(cls, type, shape, name)
        self._cached_placeholder = None
        return self

    def build_placeholder(self, prefix=''):
        """
        Build a tf.placeholder from the metadata, with an optional prefix.

        Args:
            prefix(str): the name of the placeholder will be ``prefix + self.name``

        Returns:
            tf.Tensor:
        """
        with tf.name_scope(None):   # clear any name scope it might get called in
            ret = tf.placeholder(
                self.type, shape=self.shape,
                name=prefix + self.name)
        if prefix == '' and self._cached_placeholder is None:
            self._cached_placeholder = ret  # cached_placeholder only caches the prefix='' case
        return ret

    # cannot memoize here, because InputDesc is hashed by its fields.
    def build_placeholder_reuse(self):
        """
        Build a tf.placeholder from the metadata, or return an old one.

        Returns:
            tf.Tensor:
        """
        if self._cached_placeholder is not None:
            return self._cached_placeholder
        return self.build_placeholder()

    @staticmethod
    def from_tensor(t):
        return InputDesc(
            t.dtype, t.shape.as_list(), t.name[:-2])


@six.add_metaclass(ABCMeta)
class ModelDescBase(object):
    """ Base class for a model description.
    """
    @memoized
    def get_inputs_desc(self):
        """
        Returns:
            list[:class:`InputDesc`]: list of the underlying :class:`InputDesc`.
        """
        return self._get_inputs()

    @abstractmethod
    def _get_inputs(self):
        """
        :returns: a list of InputDesc
        """

    def build_graph(self, *args):
        """
        Build the whole symbolic graph.
        This is supposed to be the "tower function" when used with :class:`TowerTrainer`.
        By default it will call :meth:`_build_graph`
        with a list of input tensors.

        Args:
            args ([tf.Tensor]): tensors that matches the list of
                :class:`InputDesc` defined by ``_get_inputs``.
        """
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, InputSource):
                inputs = arg.get_input_tensors()  # remove in the future?
                log_deprecated("build_graph(InputSource)", "Call with tensors in positional args instead.")
            elif isinstance(arg, (list, tuple)):
                inputs = arg
                log_deprecated("build_graph([Tensor])", "Call with positional args instead.")
            else:
                inputs = [arg]
        else:
            inputs = args

        assert len(inputs) == len(self.get_inputs_desc()), \
            "Number of inputs passed to the graph != number of inputs defined " \
            "in ModelDesc! ({} != {})".format(len(inputs), len(self.get_inputs_desc()))
        self._build_graph(inputs)

    def _build_graph(self, inputs):
        """
        This is an old interface which takes a list of tensors, instead of positional arguments.
        """
        pass


class ModelDesc(ModelDescBase):
    """
    A ModelDesc with single cost and single optimizer.
    It contains information about InputDesc, how to get cost, and how to get optimizer.
    """

    def get_cost(self):
        """
        Return the cost tensor in the graph.

        It calls :meth:`ModelDesc._get_cost()` which by default returns
        ``self.cost``. You can override :meth:`_get_cost()` if needed.

        This function also applies the collection
        ``tf.GraphKeys.REGULARIZATION_LOSSES`` to the cost automatically.
        """
        cost = self._get_cost()
        reg_cost = regularize_cost_from_collection()
        if reg_cost is not None:
            return tf.add(cost, reg_cost, name='cost_with_regularizer')
        else:
            return cost

    def _get_cost(self, *args):
        return self.cost

    @memoized
    def get_optimizer(self):
        """
        Return the memoized optimizer returned by `_get_optimizer`.

        Users of :class:`ModelDesc` will need to implement `_get_optimizer()`,
        which will only be called once per each model.

        Returns:
            a :class:`tf.train.Optimizer` instance.
        """
        return self._get_optimizer()

    def _get_optimizer(self):
        raise NotImplementedError()

    def _build_graph_get_cost(self, *inputs):
        self.build_graph(*inputs)
        return self.get_cost()

    def _build_graph_get_grads(self, *inputs):
        """
        Build the graph from inputs and return the grads.
        This is useful for most of the :class:`GraphBuilder` which expects such a function.

        Returns:
            [(grad, var)]
        """
        ctx = get_current_tower_context()
        cost = self._build_graph_get_cost(*inputs)

        if ctx.has_own_variables:
            varlist = ctx.get_collection_in_tower(tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            varlist = tf.trainable_variables()
        opt = self.get_optimizer()
        grads = opt.compute_gradients(
            cost, var_list=varlist,
            gate_gradients=False, colocate_gradients_with_ops=True)
        grads = FilterNoneGrad().process(grads)
        return grads
