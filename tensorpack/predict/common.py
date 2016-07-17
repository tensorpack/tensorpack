# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from collections import namedtuple
import six
from six.moves import zip

from tensorpack.models import ModelDesc
from ..utils import logger
from ..tfutils import *
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
        :param input_var_names: a list of input variable names.
        :param input_data_mapping: deprecated. used to select `input_var_names` from the `InputVars` of the model.
        :param model: a `ModelDesc` instance
        :param output_var_names: a list of names of the output tensors to predict, the
            variables can be any computable tensor in the graph.
            Predict specific output might not require all input variables.
        :param return_input: whether to return (input, output) pair or just output. default to False.
        """
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
        self.input_var_names = kwargs.pop('input_var_names', None)
        input_mapping = kwargs.pop('input_data_mapping', None)
        if input_mapping:
            raw_vars = self.model.get_input_vars_desc()
            self.input_var_names = [raw_vars[k].name for k in input_mapping]
            logger.warn('The option `input_data_mapping` was deprecated. \
Use \'input_var_names=[{}]\' instead'.format(', '.join(self.input_var_names)))
        elif self.input_var_names is None:
            # neither options is set, assume all inputs
            raw_vars = self.model.get_input_vars_desc()
            self.input_var_names = [k.name for k in raw_vars]
        self.output_var_names = kwargs.pop('output_var_names')
        assert len(self.input_var_names), self.input_var_names
        for v in self.input_var_names: assert_type(v, six.string_types)
        assert len(self.output_var_names), self.output_var_names

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

