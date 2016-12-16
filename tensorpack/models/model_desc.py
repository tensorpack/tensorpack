#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import re
import tensorflow as tf
from collections import namedtuple
import inspect
import pickle
import six

from ..utils import logger, INPUT_VARS_KEY
from ..tfutils.common import get_tensors_by_names
from ..tfutils.gradproc import CheckGradient
from ..tfutils.tower import get_current_tower_context

__all__ = ['ModelDesc', 'InputVar', 'ModelFromMetaGraph' ]

#_InputVar = namedtuple('InputVar', ['type', 'shape', 'name', 'sparse'])
class InputVar(object):
    def __init__(self, type, shape, name, sparse=False):
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

        :returns: the list of raw input vars in the graph
        """
        if hasattr(self, 'reuse_input_vars'):
            return self.reuse_input_vars
        ret = self.build_placeholders()
        self.reuse_input_vars = ret
        return ret

    # alias
    get_reuse_placehdrs = get_input_vars

    def build_placeholders(self, prefix=''):
        """ build placeholders with optional prefix, for each InputVar
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
        """ return a list of `InputVar` instance"""
        return self._get_input_vars()

    @abstractmethod
    def _get_input_vars(self):
        """:returns: a list of InputVar """

    def build_graph(self, model_inputs):
        """
        Setup the whole graph.

        :param model_inputs: a list of input variable in the graph.
        :param is_training: a boolean
        :returns: the cost to minimize. a scalar variable
        """
        if len(inspect.getargspec(self._build_graph).args) == 3:
            logger.warn("[DEPRECATED] _build_graph(self, input_vars, is_training) is deprecated! \
Use _build_graph(self, input_vars) and get_current_tower_context().is_training instead.")
            self._build_graph(model_inputs, get_current_tower_context().is_training)
        else:
            self._build_graph(model_inputs)

    @abstractmethod
    def _build_graph(self, inputs):
        pass

    def get_cost(self):
        return self._get_cost()

    def _get_cost(self, *args):
        return self.cost

    def get_gradient_processor(self):
        """ Return a list of GradientProcessor. They will be executed in order"""
        return [#SummaryGradient(),
                CheckGradient()
                ]

class ModelFromMetaGraph(ModelDesc):
    """
    Load the whole exact TF graph from a saved meta_graph.
    Only useful for inference.
    """
    def __init__(self, filename):
        tf.train.import_meta_graph(filename)
        all_coll = tf.get_default_graph().get_all_collection_keys()
        for k in [INPUT_VARS_KEY, tf.GraphKeys.TRAINABLE_VARIABLES,
                tf.GraphKeys().VARIABLES]:
            assert k in all_coll, \
                    "Collection {} not found in metagraph!".format(k)

    def _get_input_vars(self):
        col = tf.get_collection(INPUT_VARS_KEY)
        col = [InputVar.loads(v) for v in col]
        return col

    def _build_graph(self, _, __):
        """ Do nothing. Graph was imported already """
        pass
