#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta, abstractproperty
import tensorflow as tf
from ..tfutils import get_vars_by_names


class PredictorBase(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def session(self):
        """ return the session the predictor is running on"""
        pass

    def __call__(self, dp):
        assert len(dp) == len(self.input_var_names), \
            "{} != {}".format(len(dp), len(self.input_var_names))
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


class OfflinePredictor(PredictorBase):
    """ Build a predictor from a given config, in an independent graph"""
    def __init__(self, config):
        self.graph = tf.Graph()
        with self.graph.as_default():
            input_vars = config.model.get_input_vars()
            config.model._build_graph(input_vars, False)

            self.input_var_names = config.input_var_names
            self.output_var_names = config.output_var_names
            self.return_input = config.return_input

            self.input_vars = get_vars_by_names(self.input_var_names)
            self.output_vars = get_vars_by_names(self.output_var_names)

            sess = tf.Session(config=config.session_config)
            config.session_init.init(sess)
            self._session = sess

    @property
    def session(self):
        return self._session

    def _do_call(self, dp):
        feed = dict(zip(self.input_vars, dp))
        output = self.session.run(self.output_vars, feed_dict=feed)
        return output
