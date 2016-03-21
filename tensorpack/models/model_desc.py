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
    __metaclass__ = ABCMeta


    def __init__(self):
        pass

    def get_input_vars(self):
        """
        return the list of raw input vars in the graph
        if reuse=True, results will be cached, to avoid creating the same variable
        """
        input_vars = self._get_input_vars()
        ret = []
        for v in input_vars:
            ret.append(tf.placeholder(v.type, shape=v.shape, name=v.name))
        return ret

    def reuse_input_vars(self):
        """ find input_vars in default graph"""
        input_var_names = [k.name for k in self._get_input_vars()]
        g = tf.get_default_graph()
        return [g.get_tensor_by_name(name + ":0") for name in input_var_names]

    @abstractmethod
    def _get_input_vars(self):
        """
        return the list of input vars in the graph
        """
        pass

    # TODO move this to QueueInputTrainer
    def get_input_queue(self, input_vars):
        """
        return the queue for input. the dequeued elements will be fed to self.get_cost
        if queue is None, datapoints from dataflow will be fed to the graph directly.
        when running with multiGPU, queue cannot be None
        """
        assert input_vars is not None
        return tf.FIFOQueue(100, [x.dtype for x in input_vars], name='input_queue')

    def get_cost(self, input_vars, is_training):
        assert type(is_training) == bool
        return self._get_cost(input_vars, is_training)

    @abstractmethod
    def _get_cost(self, input_vars, is_training):
        """
        Args:
            input_vars: a list of input variable in the graph
            e.g.: [image_var, label_var] with:
                image_var: bx28x28
                label_var: bx1 integer
            is_training: a python bool variable
        Returns:
            the cost to minimize. scalar variable
        """

    def get_gradient_processor(self):
        """ Return a list of GradientProcessor. They will be executed in order"""
        return [CheckGradient(), SummaryGradient()]
