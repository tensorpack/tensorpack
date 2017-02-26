#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta
import tensorflow as tf
import six

from ..utils import logger
from ..utils.develop import deprecated
from ..utils.argtools import memoized
from ..utils.naming import SUMMARY_BACKUP_KEYS
from ..tfutils import get_tensors_by_names, TowerContext, get_op_tensor_name
from ..tfutils.collection import freeze_collection

__all__ = ['PredictorBase', 'AsyncPredictorBase',
           'OnlinePredictor', 'OfflinePredictor',
           'get_predict_func',
           'PredictorTowerBuilder',
           'build_prediction_graph',
           ]


@six.add_metaclass(ABCMeta)
class PredictorBase(object):
    """
    Base class for all predictors.

    Attributes:
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
        Args:
            dp: input datapoint.  must have the same length as input_names
        Returns:
            output as defined by the config
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
    """ A predictor which directly use an existing session and given tensors.
    """

    def __init__(self, input_tensors, output_tensors,
                 return_input=False, sess=None):
        """
        Args:
            input_tensors (list): list of names.
            output_tensors (list): list of names.
            return_input (bool): same as :attr:`PredictorBase.return_input`.
            sess (tf.Session): the session this predictor runs in. If None,
                will use the default session.
        """
        self.return_input = return_input
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.sess = sess

    def _do_call(self, dp):
        assert len(dp) == len(self.input_tensors), \
            "{} != {}".format(len(dp), len(self.input_tensors))
        feed = dict(zip(self.input_tensors, dp))
        if self.sess is None:
            sess = tf.get_default_session()
        else:
            sess = self.sess
        output = sess.run(self.output_tensors, feed_dict=feed)
        return output


class OfflinePredictor(OnlinePredictor):
    """ A predictor built from a given config.
        A sinlge-tower model will be built without any prefix. """

    def __init__(self, config):
        """
        Args:
            config (PredictConfig): the config to use.
        """
        self.graph = config._maybe_create_graph()
        with self.graph.as_default():
            input_placehdrs = config.model.get_reused_placehdrs()
            with TowerContext('', False):
                config.model.build_graph(input_placehdrs)

            input_tensors = get_tensors_by_names(config.input_names)
            output_tensors = get_tensors_by_names(config.output_names)

            sess = config.session_creator.create_session()
            config.session_init.init(sess)
            super(OfflinePredictor, self).__init__(
                input_tensors, output_tensors, config.return_input, sess)


@deprecated("Use OfflinePredictor instead!", "2017-05-20")
def get_predict_func(config):
    """
    Equivalent to ``OfflinePredictor(config)``.
    """
    return OfflinePredictor(config)


class PredictorTowerBuilder(object):
    """
    A builder which caches the predictor tower it has built.
    """
    def __init__(self, build_tower_fn, prefix=''):
        """
        Args:
            build_tower_fn: a function that will be called inside each tower, taking tower id as the argument.
            prefix: an extra prefix in tower name. The final tower prefix will be
                determined by :meth:`TowerContext.get_predict_tower_name`.
        """
        self._fn = build_tower_fn
        self._prefix = prefix

    @memoized
    def build(self, tower):
        """
        Args:
            tower (int): the tower will be built on device '/gpu:{tower}', or
                '/cpu:0' if tower is -1.
        """
        towername = TowerContext.get_predict_tower_name(tower, self._prefix)
        if self._prefix:
            msg = "Building predictor graph {} on gpu={} with prefix='{}' ...".format(
                towername, tower, self._prefix)
        else:
            msg = "Building predictor graph {} on gpu={} ...".format(towername, tower)
        logger.info(msg)
        # No matter where this get called, clear any existing name scope.
        with tf.name_scope(None),   \
                freeze_collection(SUMMARY_BACKUP_KEYS), \
                tf.device('/gpu:{}'.format(tower) if tower >= 0 else '/cpu:0'), \
                TowerContext(towername, is_training=False):
            self._fn(tower)

    @staticmethod
    def get_tensors_maybe_in_tower(placeholder_names, names, k, prefix=''):
        """
        Args:
            placeholders (list): A list of __op__ name.
        """
        def maybe_inside_tower(name):
            name = get_op_tensor_name(name)[0]
            if name in placeholder_names:
                return name
            else:
                # if the name is not a placeholder, use it's name in each tower
                return TowerContext.get_predict_tower_name(k, prefix) + '/' + name
        names = list(map(maybe_inside_tower, names))
        tensors = get_tensors_by_names(names)
        return tensors


def build_prediction_graph(build_tower_fn, towers=[0], prefix=''):
    """
    Execute `build_tower_fn` on each tower.
    Just a wrapper on :class:`PredictorTowerBuilder` to run on several towers
    together.
    """
    builder = PredictorTowerBuilder(build_tower_fn, prefix)
    for idx, t in enumerate(towers):
        # The first variable scope may or may not reuse (depending on the existing
        # context), but the rest have to reuse.
        with tf.variable_scope(
                tf.get_variable_scope(), reuse=True if idx > 0 else None):
            builder.build(t)
