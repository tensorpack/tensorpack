#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import re
import tensorflow as tf
from collections import namedtuple
import inspect

from ..utils import logger, INPUT_VARS_KEY
from ..tfutils import *

__all__ = ['ModelDesc', 'InputVar', 'ModelFromMetaGraph',
        'get_current_tower_context', 'TowerContext']

InputVar = namedtuple('InputVar', ['type', 'shape', 'name'])

_CurrentTowerContext = None

class TowerContext(object):
    def __init__(self, tower_name, is_training=None):
        """ tower_name: 'tower0', 'towerp0', or '' """
        self._name = tower_name
        if is_training is None:
            is_training = not self._name.startswith('towerp')
        self._is_training = is_training

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
    def name(self):
        return self._name

    def get_variable_on_tower(self, *args, **kwargs):
        """
        Get a variable for this tower specifically, without reusing.
        Tensorflow doesn't allow reuse=False scope under a
        reuse=True scope. This method provides a work around.
        See https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html#basics-of-tfvariable-scope

        :param args, kwargs: same as tf.get_variable()
        """
        with tf.variable_scope(self._name) as scope:
            with tf.variable_scope(scope, reuse=False):
                scope = tf.get_variable_scope()
                assert scope.reuse == False
                return tf.get_variable(*args, **kwargs)

    def find_tensor_in_main_tower(self, graph, name):
        if self.is_main_tower:
            return graph.get_tensor_by_name(name)
        if name.startswith('towerp'):
            newname = re.sub('towerp[0-9]+/', '', name)
            try:
                return graph.get_tensor_by_name(newname)
            except KeyError:
                newname = re.sub('towerp[0-9]+/', 'tower0/', name)
                return graph.get_tensor_by_name(newname)

    def __enter__(self):
        global _CurrentTowerContext
        assert _CurrentTowerContext is None, \
                "Nesting TowerContext!"
        _CurrentTowerContext = self
        if len(self._name):
            self._scope = tf.name_scope(self._name)
            return self._scope.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CurrentTowerContext
        _CurrentTowerContext = None
        if len(self._name):
            self._scope.__exit__(exc_type, exc_val, exc_tb)
        return False

def get_current_tower_context():
    global _CurrentTowerContext
    return _CurrentTowerContext

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

    def build_graph(self, model_inputs):
        """
        Setup the whole graph.

        :param model_inputs: a list of input variable in the graph.
        :param is_training: a boolean
        :returns: the cost to minimize. a scalar variable
        """
        if len(inspect.getargspec(self._build_graph).args) == 3:
            logger.warn("_build_graph(self, input_vars, is_training) is deprecated! \
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
