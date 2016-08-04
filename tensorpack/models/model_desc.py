#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import namedtuple

from ..utils import logger, INPUT_VARS_KEY
from ..tfutils import *

__all__ = ['ModelDesc', 'InputVar', 'ModelFromMetaGraph']

InputVar = namedtuple('InputVar', ['type', 'shape', 'name'])

class ModelDesc(object):
    """ Base class for a model description """
    __metaclass__ = ABCMeta

    def get_input_vars(self):
        """
        Create or return (if already created) raw input TF placeholder vars in the graph.

        :returns: the list of raw input vars in the graph
        """
        try:
            return self.reuse_input_vars()
        except KeyError:
            pass
        input_vars = self._get_input_vars()
        ret = []
        for v in input_vars:
            ret.append(tf.placeholder(v.type, shape=v.shape, name=v.name))
        for v in ret:
            tf.add_to_collection(INPUT_VARS_KEY, v)
        return ret

    def reuse_input_vars(self):
        """ Find and return already-defined input_vars in default graph"""
        input_var_names = [k.name for k in self._get_input_vars()]
        return get_vars_by_names(input_var_names)

    def get_input_vars_desc(self):
        """ return a list of `InputVar` instance"""
        return self._get_input_vars()

    @abstractmethod
    def _get_input_vars(self):
        """:returns: a list of InputVar """

    def build_graph(self, model_inputs, is_training):
        """
        Setup the whole graph.

        :param model_inputs: a list of input variable in the graph.
        :param is_training: a boolean
        :returns: the cost to minimize. a scalar variable
        """
        self._build_graph(model_inputs, is_training)

    @abstractmethod
    def _build_graph(self, inputs, is_training):
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
                tf.GraphKeys.VARIABLES]:
            assert k in all_coll, \
                    "Collection {} not found in metagraph!".format(k)

    def get_input_vars(self):
        return tf.get_collection(INPUT_VARS_KEY)

    def _get_input_vars(self):
        raise NotImplementedError("Shouldn't call here")

    def _build_graph(self, _, __):
        """ Do nothing. Graph was imported already """
        pass
