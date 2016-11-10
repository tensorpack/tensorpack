# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from collections import namedtuple
import six
from six.moves import zip

from tensorpack.models import ModelDesc
from ..utils import logger
from ..tfutils import get_default_sess_config
from ..tfutils.sessinit import SessionInit, JustCurrentSession
from .base import OfflinePredictor

import multiprocessing

__all__ = ['PredictConfig', 'get_predict_func', 'PredictResult' ]

PredictResult = namedtuple('PredictResult', ['input', 'output'])

class PredictConfig(object):
    def __init__(self, **kwargs):
        """
        The config used by `get_predict_func`.

        :param session_init: a `utils.sessinit.SessionInit` instance to
            initialize variables of a session.
        :param model: a `ModelDesc` instance
        :param input_names: a list of input variable names.
        :param output_names: a list of names of the output tensors to predict, the
            variables can be any computable tensor in the graph.
            Predict specific output might not require all input variables.
        :param return_input: whether to return (input, output) pair or just output. default to False.
        """
        # TODO use the name "tensor" instead of "variable"
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__
        # XXX does it work? start with minimal memory, but allow growth.
        # allow_growth doesn't seem to work very well in TF.
        self.session_config = kwargs.pop('session_config', get_default_sess_config(0.4))
        self.session_init = kwargs.pop('session_init', JustCurrentSession())
        assert_type(self.session_init, SessionInit)
        self.model = kwargs.pop('model')
        assert_type(self.model, ModelDesc)

        # inputs & outputs
        # TODO add deprecated warning later
        self.input_names = kwargs.pop('input_names', None)
        if self.input_names is None:
            self.input_names = kwargs.pop('input_var_names', None)
            if self.input_names is not None:
                pass
                #logger.warn("[Deprecated] input_var_names is deprecated in PredictConfig. Use input_names instead!")
        if self.input_names is None:
            # neither options is set, assume all inputs
            raw_vars = self.model.get_input_vars_desc()
            self.input_names = [k.name for k in raw_vars]
        self.output_names = kwargs.pop('output_names', None)
        if self.output_names is None:
            self.output_names = kwargs.pop('output_var_names')
            #logger.warn("[Deprecated] output_var_names is deprecated in PredictConfig. Use output_names instead!")
        assert len(self.input_names), self.input_names
        for v in self.input_names: assert_type(v, six.string_types)
        assert len(self.output_names), self.output_names

        self.return_input = kwargs.pop('return_input', False)
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

def get_predict_func(config):
    """
    Produce a offline predictor run inside a new session.

    :param config: a `PredictConfig` instance.
    :returns: A callable predictor that takes a list of input values, and return
        a list of output values defined in ``config.output_var_names``.
    """
    return OfflinePredictor(config)

