#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import tensorflow as tf

__all__ = ['ModelDesc']

class ModelDesc(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.input_vars = None

    def get_input_vars(self):
        """
        return the list of input vars in the graph
        results will be cached, to avoid creating the same variable

        """
        if self.input_vars is None:
            self.input_vars = self._get_input_vars()
            for i in self.input_vars:
                assert isinstance(i, tf.Tensor), tf.Tensor.__class__
        return self.input_vars

    @abstractmethod
    def _get_input_vars(self):
        """
        return the list of input vars in the graph
        """

    def get_input_queue(self):
        """
        return the queue for input. the dequeued elements will be fed to self.get_cost
        if queue is None, datapoints from dataflow will be fed to the graph directly.
        when running with multiGPU, queue cannot be None
        """
        assert self.input_vars is not None
        return tf.FIFOQueue(50, [x.dtype for x in self.input_vars], name='input_queue')

    def get_cost(self, input_vars, is_training):
        assert len(input_vars) == len(self.input_vars)
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

        input_vars might be different from self.input_vars
        (inputs might go through the queue for faster input),
        but must have the same length
        """

    def get_lr_multiplier(self):
        """
        Return a list of (variable_regex: multiplier)
        """
        return []
