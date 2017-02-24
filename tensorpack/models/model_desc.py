#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import pickle
import six

from ..utils import logger
from ..utils.naming import INPUTS_KEY
from ..utils.develop import deprecated, log_deprecated
from ..utils.argtools import memoized
from ..tfutils.modelutils import apply_slim_collections

__all__ = ['InputDesc', 'InputVar', 'ModelDesc', 'ModelFromMetaGraph']

# TODO "variable" is not the right name to use for input here.


class InputDesc(object):
    """ Store metadata about input placeholders. """
    def __init__(self, type, shape, name, sparse=False):
        """
        Args:
            type: tf type of the tensor.
            shape (list):
            name (str):
            sparse (bool): whether to use ``tf.sparse_placeholder``.
        """
        self.type = type
        self.shape = shape
        self.name = name
        self.sparse = sparse

    def dumps(self):
        return pickle.dumps(self)

    @staticmethod
    def loads(buf):
        return pickle.loads(buf)


class InputVar(InputDesc):
    def __init__(self, *args, **kwargs):
        logger.warn("[Deprecated] InputVar was renamed to InputDesc!")
        super(InputVar, self).__init__(*args, **kwargs)


@six.add_metaclass(ABCMeta)
class ModelDesc(object):
    """ Base class for a model description """

# inputs:
    @memoized
    def get_reused_placehdrs(self):
        """
        Create or return (if already created) raw input TF placeholders in the graph.

        Returns:
            list[tf.Tensor]: the list of input placeholders in the graph.
        """
        return self.build_placeholders()

    @deprecated("Use get_reused_placehdrs() instead.", "2017-04-11")
    def get_input_vars(self):
        # this wasn't a public API anyway
        return self.get_reused_placehdrs()

    def build_placeholders(self, prefix=''):
        """
        For each InputDesc, create new placeholders with optional prefix and
        return them. Useful when building new towers.

        Returns:
            list[tf.Tensor]: the list of built placeholders.
        """
        input_vars = self._get_inputs()
        for v in input_vars:
            tf.add_to_collection(INPUTS_KEY, v.dumps())
        ret = []
        with tf.name_scope(None):   # clear any name scope it might get called in
            for v in input_vars:
                placehdr_f = tf.placeholder if not v.sparse else tf.sparse_placeholder
                ret.append(placehdr_f(
                    v.type, shape=v.shape,
                    name=prefix + v.name))
        return ret

    def get_inputs_desc(self):
        """
        Returns:
            list[:class:`InputDesc`]: list of the underlying :class:`InputDesc`.
        """
        return self._get_inputs()

    def _get_inputs(self):  # this is a better name than _get_input_vars
        """
        :returns: a list of InputDesc
        """
        log_deprecated("", "_get_input_vars() was renamed to _get_inputs().", "2017-04-11")
        return self._get_input_vars()

    def _get_input_vars(self):  # keep backward compatibility
        raise NotImplementedError()

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

        This function also applies tfslim collections to the cost automatically,
        including ``tf.GraphKeys.REGULARIZATION_LOSSES`` and ``tf.GraphKeys.UPDATE_OPS``.
        This is because slim users would expect the regularizer being automatically applied once used in slim layers.
        """
        cost = self._get_cost()
        return apply_slim_collections(cost)

    def _get_cost(self, *args):
        return self.cost

    @memoized
    def get_optimizer(self):
        """
        Return the optimizer used in the task.
        Used by some of the tensorpack :class:`Trainer` which only uses a single optimizer.
        You can ignore this method if you use your own trainer with more than one optimizers.

        Users of :class:`ModelDesc` will need to implement `_get_optimizer()`,
        which will only be called once per each model.

        Returns:
            a :class:`tf.train.Optimizer` instance.
        """
        return self._get_optimizer()

    def _get_optimizer(self):
        raise NotImplementedError()

    def get_gradient_processor(self):
        return []


class ModelFromMetaGraph(ModelDesc):
    """
    Load the exact TF graph from a saved meta_graph.
    Only useful for inference.
    """

    # TODO this class may not be functional anymore.

    def __init__(self, filename):
        """
        Args:
            filename (str): file name of the saved meta graph.
        """
        tf.train.import_meta_graph(filename)
        all_coll = tf.get_default_graph().get_all_collection_keys()
        for k in [INPUTS_KEY, tf.GraphKeys.TRAINABLE_VARIABLES,
                  tf.GraphKeys.GLOBAL_VARIABLES]:
            assert k in all_coll, \
                "Collection {} not found in metagraph!".format(k)

    def _get_inputs(self):
        col = tf.get_collection(INPUTS_KEY)
        col = [InputDesc.loads(v) for v in col]
        return col

    def _build_graph(self, _, __):
        """ Do nothing. Graph was imported already """
        pass
