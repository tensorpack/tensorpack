#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta
import tensorflow as tf
import six

from ..utils.naming import PREDICT_TOWER
from ..utils import logger
from ..tfutils import get_tensors_by_names, TowerContext

__all__ = ['PredictorBase', 'AsyncPredictorBase',
           'OnlinePredictor', 'OfflinePredictor',
           'MultiTowerOfflinePredictor', 'build_multi_tower_prediction_graph',
           'DataParallelOfflinePredictor']


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


def build_multi_tower_prediction_graph(build_tower_fn, towers):
    """
    Args:
        build_tower_fn: a function that will be called inside each tower,
            taking tower id as the argument.
        towers: a list of relative GPU id.
    """
    for k in towers:
        logger.info(
            "Building graph for predictor tower {}...".format(k))
        with tf.device('/gpu:{}'.format(k) if k >= 0 else '/cpu:0'), \
                TowerContext('{}{}'.format(PREDICT_TOWER, k)):
            build_tower_fn(k)
            tf.get_variable_scope().reuse_variables()


class MultiTowerOfflinePredictor(OnlinePredictor):
    """ A multi-tower multi-GPU predictor. """

    def __init__(self, config, towers):
        """
        Args:
            config (PredictConfig): the config to use.
            towers: a list of relative GPU id.
        """
        self.graph = tf.Graph()
        self.predictors = []
        with self.graph.as_default():
            # TODO backup summary keys?
            def fn(_):
                config.model.build_graph(config.model.get_input_vars())
            build_multi_tower_prediction_graph(fn, towers)

            self.sess = tf.Session(config=config.session_config)
            config.session_init.init(self.sess)

            input_vars = get_tensors_by_names(config.input_names)

            for k in towers:
                output_vars = get_tensors_by_names(
                    ['{}{}/'.format(PREDICT_TOWER, k) + n
                     for n in config.output_names])
                self.predictors.append(OnlinePredictor(
                    self.sess, input_vars, output_vars, config.return_input))

    def _do_call(self, dp):
        # use the first tower for compatible PredictorBase interface
        return self.predictors[0]._do_call(dp)

    def get_predictors(self, n):
        """
        Returns:
            PredictorBase: the nth predictor on the nth GPU.
        """
        return [self.predictors[k % len(self.predictors)] for k in range(n)]


class DataParallelOfflinePredictor(OnlinePredictor):
    """ A data-parallel predictor.
    It runs different towers in parallel.
    """

    def __init__(self, config, towers):
        """
        Args:
            config (PredictConfig): the config to use.
            towers: a list of relative GPU id.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            sess = tf.Session(config=config.session_config)
            input_var_names = []
            output_vars = []
            for k in towers:
                towername = PREDICT_TOWER + str(k)
                input_vars = config.model.build_placeholders(
                    prefix=towername + '-')
                logger.info(
                    "Building graph for predictor tower {}...".format(k))
                with tf.device('/gpu:{}'.format(k) if k >= 0 else '/cpu:0'), \
                        TowerContext(towername, is_training=False):
                    config.model.build_graph(input_vars)
                    tf.get_variable_scope().reuse_variables()
                input_var_names.extend([k.name for k in input_vars])
                output_vars.extend(get_tensors_by_names(
                    [towername + '/' + n
                     for n in config.output_names]))

            input_vars = get_tensors_by_names(input_var_names)
            config.session_init.init(sess)
            super(DataParallelOfflinePredictor, self).__init__(
                sess, input_vars, output_vars, config.return_input)
