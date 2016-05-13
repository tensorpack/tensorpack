#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import namedtuple

from ..tfutils import *

__all__ = ['ModelDesc', 'InputVar']

InputVar = namedtuple('InputVar', ['type', 'shape', 'name'])

class ModelDesc(object):
    """ Base class for a model description """
    __metaclass__ = ABCMeta

    def get_input_vars(self):
        """
        Create or return (if already created) input TF vars in the graph.

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
        return ret

    def reuse_input_vars(self):
        """ Find and return already-defined input_vars in default graph"""
        input_var_names = [k.name for k in self._get_input_vars()]
        g = tf.get_default_graph()
        return [g.get_tensor_by_name(name + ":0") for name in input_var_names]

    @abstractmethod
    def _get_input_vars(self):
        """:returns: a list of InputVar """

    def get_cost(self, input_vars, is_training):
        """
        :param input_vars: a list of input variable in the graph
            e.g.: [image_var, label_var] with:

            * image_var: bx28x28
            * label_var: bx1 integer
        :param is_training: a boolean
        :returns: the cost to minimize. a scalar variable
        """
        assert type(is_training) == bool
        return self._get_cost(input_vars, is_training)

    @abstractmethod
    def _get_cost(self, input_vars, is_training):
        pass

    def get_gradient_processor(self):
        """ Return a list of GradientProcessor. They will be executed in order"""
        return [CheckGradient()]#, SummaryGradient()]

