# -*- coding: UTF-8 -*-
# File: config.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import six

from tensorpack.models import ModelDesc
from ..tfutils import get_default_sess_config
from ..tfutils.sessinit import SessionInit, JustCurrentSession

__all__ = ['PredictConfig']


class PredictConfig(object):
    def __init__(self, **kwargs):
        """
        Args:
            session_init (SessionInit): how to initialize variables of the session.
            model (ModelDesc): the model to use.
            input_names (list): a list of input tensor names.
            output_names (list): a list of names of the output tensors to predict, the
                tensors can be any computable tensor in the graph.
            return_input: same as in :attr:`PredictorBase.return_input`.
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
                # logger.warn("[Deprecated] input_var_names is deprecated in PredictConfig. Use input_names instead!")
        if self.input_names is None:
            # neither options is set, assume all inputs
            raw_vars = self.model.get_input_vars_desc()
            self.input_names = [k.name for k in raw_vars]
        self.output_names = kwargs.pop('output_names', None)
        if self.output_names is None:
            self.output_names = kwargs.pop('output_var_names')
            # logger.warn("[Deprecated] output_var_names is deprecated in PredictConfig. Use output_names instead!")
        assert len(self.input_names), self.input_names
        for v in self.input_names:
            assert_type(v, six.string_types)
        assert len(self.output_names), self.output_names

        self.return_input = kwargs.pop('return_input', False)
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))
