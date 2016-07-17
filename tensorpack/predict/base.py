#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta, abstractproperty
import tensorflow as tf
from ..tfutils import get_vars_by_names

__all__ = ['OnlinePredictor', 'OfflinePredictor']


class PredictorBase(object):
    __metaclass__ = ABCMeta
    """
    Property:
    session
    return_input
    """

    def __call__(self, dp):
        output = self._do_call(dp)
        if self.return_input:
            return (dp, output)
        else:
            return output

    @abstractmethod
    def _do_call(self, dp):
        """
        :param dp: input datapoint.  must have the same length as input_var_names
        :return: output as defined by the config
        """
        pass

class OnlinePredictor(PredictorBase):
    def __init__(self, sess, input_vars, output_vars, return_input=False):
        self.session = sess
        self.return_input = return_input

        self.input_vars = input_vars
        self.output_vars = output_vars

    def _do_call(self, dp):
        assert len(dp) == len(self.input_vars), \
            "{} != {}".format(len(dp), len(self.input_vars))
        feed = dict(zip(self.input_vars, dp))
        output = self.session.run(self.output_vars, feed_dict=feed)
        return output


class OfflinePredictor(OnlinePredictor):
    """ Build a predictor from a given config, in an independent graph"""
    def __init__(self, config):
        self.graph = tf.Graph()
        with self.graph.as_default():
            input_vars = config.model.get_input_vars()
            config.model._build_graph(input_vars, False)

            input_vars = get_vars_by_names(config.input_var_names)
            output_vars = get_vars_by_names(config.output_var_names)

            sess = tf.Session(config=config.session_config)
            config.session_init.init(sess)
            super(OfflinePredictor, self).__init__(
                    sess, input_vars, output_vars, config.return_input)
