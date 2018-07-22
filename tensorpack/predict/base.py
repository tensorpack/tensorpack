# -*- coding: utf-8 -*-
# File: base.py


from abc import abstractmethod, ABCMeta
import tensorflow as tf
import six

from ..tfutils.common import get_tensors_by_names
from ..tfutils.tower import PredictTowerContext
from ..input_source import PlaceholderInput
from ..utils.develop import log_deprecated
from ..utils.utils import execute_only_once

__all__ = ['PredictorBase', 'AsyncPredictorBase',
           'OnlinePredictor', 'OfflinePredictor',
           ]


@six.add_metaclass(ABCMeta)
class PredictorBase(object):
    """
    Base class for all predictors.

    Attributes:
        return_input (bool): whether the call will also return (inputs, outputs)
            or just outputs
    """

    def __call__(self, *args):
        """
        Call the predictor on some inputs.

        Example:
            When you have a predictor defined with two inputs, call it with:

            .. code-block:: python

                predictor(e1, e2)
        """
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            dp = args[0]    # backward-compatibility
            if execute_only_once():
                log_deprecated(
                    "Calling a predictor with one datapoint",
                    "Call it with positional arguments instead!",
                    "2018-3-1")
        else:
            dp = args
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

    ACCEPT_OPTIONS = False

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

        if sess is not None:
            self._callable = sess.make_callable(
                fetches=output_tensors,
                feed_list=input_tensors,
                accept_options=self.ACCEPT_OPTIONS)
        else:
            self._callable = None

    def _do_call(self, dp):
        assert len(dp) == len(self.input_tensors), \
            "{} != {}".format(len(dp), len(self.input_tensors))
        if self.sess is None:
            self.sess = tf.get_default_session()

        if self._callable is None:
            self._callable = self.sess.make_callable(
                fetches=self.output_tensors,
                feed_list=self.input_tensors,
                accept_options=self.ACCEPT_OPTIONS)
        # run_metadata = tf.RunMetadata()
        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        return self._callable(*dp)


class OfflinePredictor(OnlinePredictor):
    """ A predictor built from a given config.
        A single-tower model will be built without any prefix. """

    def __init__(self, config):
        """
        Args:
            config (PredictConfig): the config to use.
        """
        self.graph = config._maybe_create_graph()
        with self.graph.as_default():
            input = PlaceholderInput()
            input.setup(config.inputs_desc)
            with PredictTowerContext(''):
                config.tower_func(*input.get_input_tensors())

            input_tensors = get_tensors_by_names(config.input_names)
            output_tensors = get_tensors_by_names(config.output_names)

            config.session_init._setup_graph()
            sess = config.session_creator.create_session()
            config.session_init._run_init(sess)
            super(OfflinePredictor, self).__init__(
                input_tensors, output_tensors, config.return_input, sess)
