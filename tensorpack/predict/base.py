#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta
import tensorflow as tf
import six

from ..tfutils.common import get_tensors_by_names
from ..tfutils.tower import TowerContext
from ..graph_builder.input_source import PlaceholderInput

__all__ = ['PredictorBase', 'AsyncPredictorBase',
           'OnlinePredictor', 'OfflinePredictor',
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
                will use the default session at the first call.
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
            self.sess = tf.get_default_session()
        output = self.sess.run(self.output_tensors, feed_dict=feed)
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
            input = PlaceholderInput()
            input.setup(config.model.get_inputs_desc())
            with TowerContext('', is_training=False):
                config.model.build_graph(input)

            input_tensors = get_tensors_by_names(config.input_names)
            output_tensors = get_tensors_by_names(config.output_names)

            sess = config.session_creator.create_session()
            config.session_init.init(sess)
            super(OfflinePredictor, self).__init__(
                input_tensors, output_tensors, config.return_input, sess)
