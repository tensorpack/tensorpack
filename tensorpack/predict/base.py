#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta
import tensorflow as tf
import six

from ..utils import logger
from ..tfutils import get_tensors_by_names, TowerContext

__all__ = ['PredictorBase', 'AsyncPredictorBase',
           'OnlinePredictor', 'OfflinePredictor',
           'get_predict_func',
           'build_prediction_graph',
           ]


@six.add_metaclass(ABCMeta)
class PredictorBase(object):
    """
    Base class for all predictors.

    Attributes:
        session (tf.Session):
        return_input (bool): whether the call will also return (inputs, outputs)
            or just outpus
    """

    def __call__(self, *args):
        """
        Call the predictor on some inputs.

        If ``len(args) == 1``, assume ``args[0]`` is a datapoint (a list).
        otherwise, assume ``args`` is a datapoinnt

        Examples:
            When you have a predictor which takes a datapoint [e1, e2], you
            can call it in two ways:

            .. code-block:: python

                predictor(e1, e2)
                predictor([e1, e2])
        """
        if len(args) != 1:
            dp = args
        else:
            dp = args[0]
        output = self._do_call(dp)
        if self.return_input:
            return (dp, output)
        else:
            return output

    @abstractmethod
    def _do_call(self, dp):
        """
        :param dp: input datapoint.  must have the same length as input_names
        :return: output as defined by the config
        """


class AsyncPredictorBase(PredictorBase):
    """ Base class for all async predictors. """

    @abstractmethod
    def put_task(self, dp, callback=None):
        """
        Args:
            dp (list): A datapoint as inputs. It could be either batched or not
                batched depending on the predictor implementation).
            callback: a thread-safe callback to get called with
                either outputs or (inputs, outputs).
        Returns:
            concurrent.futures.Future: a Future of results
        """

    @abstractmethod
    def start(self):
        """ Start workers """

    def _do_call(self, dp):
        assert six.PY3, "With Python2, sync methods not available for async predictor"
        fut = self.put_task(dp)
        # in Tornado, Future.result() doesn't wait
        return fut.result()


class OnlinePredictor(PredictorBase):
    """ A predictor which directly use an existing session. """

    def __init__(self, sess, input_tensors, output_tensors, return_input=False):
        """
        Args:
            sess (tf.Session): an existing session.
            input_tensors (list): list of names.
            output_tensors (list): list of names.
            return_input (bool): same as :attr:`PredictorBase.return_input`.
        """
        self.session = sess
        self.return_input = return_input

        self.input_tensors = input_tensors
        self.output_tensors = output_tensors

    def _do_call(self, dp):
        assert len(dp) == len(self.input_tensors), \
            "{} != {}".format(len(dp), len(self.input_tensors))
        feed = dict(zip(self.input_tensors, dp))
        output = self.session.run(self.output_tensors, feed_dict=feed)
        return output


class OfflinePredictor(OnlinePredictor):
    """ A predictor built from a given config, in a new graph. """

    def __init__(self, config):
        """
        Args:
            config (PredictConfig): the config to use.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            input_placehdrs = config.model.get_input_vars()
            with TowerContext('', False):
                config.model.build_graph(input_placehdrs)

            input_vars = get_tensors_by_names(config.input_names)
            output_vars = get_tensors_by_names(config.output_names)

            sess = tf.Session(config=config.session_config)
            config.session_init.init(sess)
            super(OfflinePredictor, self).__init__(
                sess, input_vars, output_vars, config.return_input)


def get_predict_func(config):
    """
    Equivalent to ``OfflinePredictor(config)``.
    """
    return OfflinePredictor(config)


def build_prediction_graph(build_tower_fn, towers=[0], prefix=''):
    """
    Args:
        build_tower_fn: a function that will be called inside each tower,
            taking tower id as the argument.
        towers: a list of relative GPU id.
        prefix: an extra prefix in tower name. The final tower prefix will be
            determined by :meth:`TowerContext.get_predict_tower_name`.
    """
    for idx, k in enumerate(towers):
        logger.info(
            "Building prediction graph for towerid={} with prefix='{}' ...".format(k, prefix))
        towername = TowerContext.get_predict_tower_name(prefix, k)
        with tf.device('/gpu:{}'.format(k) if k >= 0 else '/cpu:0'), \
                TowerContext(towername, is_training=False), \
                tf.variable_scope(tf.get_variable_scope(),
                                  reuse=True if idx > 0 else None):
            build_tower_fn(k)
