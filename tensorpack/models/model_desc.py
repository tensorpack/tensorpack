#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import pickle
import six

from ..utils import logger, INPUT_VARS_KEY
from ..tfutils.gradproc import CheckGradient
from ..tfutils.summary import add_moving_summary
from ..tfutils.tower import get_current_tower_context

__all__ = ['ModelDesc', 'InputVar', 'ModelFromMetaGraph']


# TODO "variable" is not a right name to use across this file.

class InputVar(object):
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


@six.add_metaclass(ABCMeta)
class ModelDesc(object):
    """ Base class for a model description """

    def get_input_vars(self):
        """
        Create or return (if already created) raw input TF placeholder vars in the graph.

        Returns:
            list[tf.Tensor]: the list of input placeholders in the graph.
        """
        if hasattr(self, 'reuse_input_vars'):
            return self.reuse_input_vars
        ret = self.build_placeholders()
        self.reuse_input_vars = ret
        return ret

    # alias
    get_reuse_placehdrs = get_input_vars

    def build_placeholders(self, prefix=''):
        """
        For each InputVar, create new placeholders with optional prefix and
        return them. Useful when building new towers.

        Returns:
            list[tf.Tensor]: the list of built placeholders.
        """
        input_vars = self._get_input_vars()
        for v in input_vars:
            tf.add_to_collection(INPUT_VARS_KEY, v.dumps())
        ret = []
        for v in input_vars:
            placehdr_f = tf.placeholder if not v.sparse else tf.sparse_placeholder
            ret.append(placehdr_f(
                v.type, shape=v.shape,
                name=prefix + v.name))
        return ret

    def get_input_vars_desc(self):
        """
        Returns:
            list[:class:`InputVar`]: list of the underlying :class:`InputVar`.
        """
        return self._get_input_vars()

    def _get_input_vars(self):  # keep backward compatibility
        """
        :returns: a list of InputVar
        """
        return self._get_inputs()

    def _get_inputs(self):  # this is a better name than _get_input_vars
        raise NotImplementedError()

    def build_graph(self, model_inputs):
        """
        Build the whole symbolic graph.

        Args:
            model_inputs (list[tf.Tensor]): a list of inputs, corresponding to
                InputVars of this model.
        """
        self._build_graph(model_inputs)

    @abstractmethod
    def _build_graph(self, inputs):
        pass

    def get_cost(self):
        """
        Return the cost tensor in the graph. Called by some of the :class:`tensorpack.train.Trainer` which
        assumes single-cost models.

        This function also apply tfslim collections to the cost automatically, including
        ``tf.GraphKeys.REGULARIZATION_LOSSES`` and
        ``tf.GraphKeys.UPDATE_OPS``. This is because slim users would expect
        the regularizer being automatically applied once used in slim layers.
        """

        # the model cost so far
        cost = self._get_cost()

        regulization_losses = set(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        if len(regulization_losses) > 0:
            reg_loss = tf.add_n(list(regulization_losses), name="regularize_loss")
            cost = tf.add(reg_loss, cost, name='total_cost')
            add_moving_summary(reg_loss, cost)

        # As these batch-norm statistics quickly accumulate, there is no significant loss of accuracy
        # if only the main tower handles all batch-normalization updates, which are then shared across
        # the towers
        ctx = get_current_tower_context()
        if ctx is not None and ctx.is_main_training_tower:
            non_grad_updates = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            if non_grad_updates:
                logger.info("Apply UPDATE_OPS collection on cost.")
                with tf.control_dependencies(non_grad_updates):
                    cost = tf.identity(cost)
        return cost

    def _get_cost(self, *args):
        return self.cost

    def get_gradient_processor(self):
        """ Return a list of :class:`tensorpack.tfutils.GradientProcessor`.
            They will be executed by the trainer in the given order.
        """
        return [  # SummaryGradient(),
            CheckGradient()
        ]


class ModelFromMetaGraph(ModelDesc):
    """
    Load the exact TF graph from a saved meta_graph.
    Only useful for inference.
    """

    # TODO can this be really used for inference?

    def __init__(self, filename):
        """
        Args:
            filename (str): file name of the saved meta graph.
        """
        tf.train.import_meta_graph(filename)
        all_coll = tf.get_default_graph().get_all_collection_keys()
        for k in [INPUT_VARS_KEY, tf.GraphKeys.TRAINABLE_VARIABLES,
                  tf.GraphKeys.GLOBAL_VARIABLES]:
            assert k in all_coll, \
                "Collection {} not found in metagraph!".format(k)

    def _get_inputs(self):
        col = tf.get_collection(INPUT_VARS_KEY)
        col = [InputVar.loads(v) for v in col]
        return col

    def _build_graph(self, _, __):
        """ Do nothing. Graph was imported already """
        pass
