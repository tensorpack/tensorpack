#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
from collections import namedtuple
import tensorflow as tf
import pickle
import six

from ..utils import logger
from ..utils.argtools import memoized
from .regularize import regularize_cost_from_collection

__all__ = ['InputDesc', 'InputVar', 'ModelDesc']


class InputDesc(
        namedtuple('InputDescTuple', ['type', 'shape', 'name'])):
    """
    Metadata about an input entry point to the graph.
    This metadata can be later used to build placeholders or other types of
    input source.
    """

    def dumps(self):
        """
        Returns:
            str: serialized string
        """
        return pickle.dumps(self)

    @staticmethod
    def loads(buf):
        """
        Args:
            buf (str): serialized string

        Returns:
            InputDesc:
        """
        return pickle.loads(buf)

    def build_placeholder(self, prefix=''):
        """
        Build a tf.placeholder from the metadata, with an optional prefix.

        Args:
            prefix(str): the name of the placeholder will be ``prefix + self.name``

        Returns:
            tf.Tensor:
        """
        with tf.name_scope(None):   # clear any name scope it might get called in
            return tf.placeholder(
                self.type, shape=self.shape,
                name=prefix + self.name)

    # TODO cache results from build_placeholder, and skip it in serialization
    @memoized
    def build_placeholder_reuse(self):
        """
        Build a tf.placeholder from the metadata, or return an old one.

        Returns:
            tf.Tensor:
        """
        return self.build_placeholder()


class InputVar(InputDesc):
    def __init__(self, *args, **kwargs):
        logger.warn("[Deprecated] InputVar was renamed to InputDesc!")
        super(InputVar, self).__init__(*args, **kwargs)


@six.add_metaclass(ABCMeta)
class ModelDesc(object):
    """ Base class for a model description.
    """

# inputs:
    @memoized
    def get_reused_placehdrs(self):
        """
        Create or return (if already created) raw input TF placeholders in the graph.

        Returns:
            list[tf.Tensor]: the list of input placeholders in the graph.
        """
        return self.build_placeholders()

    def build_placeholders(self, prefix=''):
        """
        For each InputDesc, create new placeholders with optional prefix and
        return them. Useful when building new towers.

        Returns:
            list[tf.Tensor]: the list of built placeholders.
        """
        inputs = self._get_inputs()
        ret = []
        for v in inputs:
            ret.append(v.build_placeholder(prefix))
        return ret

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

    def build_graph(self, model_inputs):
        """
        Build the whole symbolic graph.

        Args:
            model_inputs (list[tf.Tensor]): a list of inputs, corresponding to
                InputDesc of this model.
        """
        self._build_graph(model_inputs)

    @abstractmethod
    def _build_graph(self, inputs):
        pass

    def get_cost(self):
        """
        Return the cost tensor in the graph.
        Used by some of the tensorpack :class:`Trainer` which assumes single-cost models.
        You can ignore this method if you use your own trainer with more than one cost.

        It calls :meth:`ModelDesc._get_cost()` which by default returns
        ``self.cost``. You can override :meth:`_get_cost()` if needed.

        This function also applies the collection
        ``tf.GraphKeys.REGULARIZATION_LOSSES`` to the cost automatically.
        """
        cost = self._get_cost()
        return tf.add(cost, regularize_cost_from_collection(),
                      name='cost_with_regularizer')

    def _get_cost(self, *args):
        return self.cost

    @memoized
    def get_optimizer(self):
        """
        Return the optimizer used in the task.
        Used by some of the tensorpack :class:`Trainer` which assume single optimizer.
        You can (and should) ignore this method if you use a custom trainer with more than one optimizers.

        Users of :class:`ModelDesc` will need to implement `_get_optimizer()`,
        which will only be called once per each model.

        Returns:
            a :class:`tf.train.Optimizer` instance.
        """
        return self._get_optimizer()

    def _get_optimizer(self):
        raise NotImplementedError()

    def get_gradient_processor(self):
        return self._get_gradient_processor()

    def _get_gradient_processor(self):
        return []
