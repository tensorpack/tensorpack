# -*- coding: UTF-8 -*-
# File: config.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import six

from ..models import ModelDesc
from ..tfutils import get_default_sess_config
from ..tfutils.sessinit import SessionInit, JustCurrentSession

__all__ = ['PredictConfig']


class PredictConfig(object):
    def __init__(self, model, session_init=None,
                 session_config=get_default_sess_config(0.4),
                 input_names=None,
                 output_names=None,
                 return_input=False):
        """
        Args:
            model (ModelDesc): the model to use.
            session_init (SessionInit): how to initialize variables of the session.
                Defaults to do nothing.
            session_config]
            input_names (list): a list of input tensor names. Defaults to all
                inputs of the model.
            output_names (list): a list of names of the output tensors to predict, the
                tensors can be any computable tensor in the graph.
            return_input: same as in :attr:`PredictorBase.return_input`.
        """
        # TODO use the name "tensor" instead of "variable"
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__
        self.model = model
        assert_type(self.model, ModelDesc)

        # XXX does it work? start with minimal memory, but allow growth.
        # allow_growth doesn't seem to work very well in TF.
        self.session_config = session_config
        if session_init is None:
            session_init = JustCurrentSession()
        self.session_init = session_init
        assert_type(self.session_init, SessionInit)

        # inputs & outputs
        self.input_names = input_names
        if self.input_names is None:
            # neither options is set, assume all inputs
            raw_vars = self.model.get_input_vars_desc()
            self.input_names = [k.name for k in raw_vars]
        self.output_names = output_names
        assert_type(self.output_names, list)
        assert_type(self.input_names, list)
        assert len(self.input_names), self.input_names
        for v in self.input_names:
            assert_type(v, six.string_types)
        assert len(self.output_names), self.output_names

        self.return_input = bool(return_input)
