# -*- coding: utf-8 -*-
# File: config.py


import six
import tensorflow as tf

from ..graph_builder import ModelDescBase
from ..tfutils import get_default_sess_config
from ..tfutils.sessinit import JustCurrentSession, SessionInit
from ..tfutils.tower import TowerFuncWrapper
from ..utils import logger

__all__ = ['PredictConfig']


class PredictConfig(object):
    def __init__(self,
                 model=None,
                 tower_func=None,
                 inputs_desc=None,

                 input_names=None,
                 output_names=None,

                 session_creator=None,
                 session_init=None,
                 return_input=False,
                 create_graph=True,
                 ):
        """
        You need to set either `model`, or `inputs_desc` plus `tower_func`.
        They are needed to construct the graph.
        You'll also have to set `output_names` as it does not have a default.

        Example:

        .. code-block:: python

            config = PredictConfig(model=my_model,
                                   inputs_names=['image'],
                                   output_names=['linear/output', 'prediction'])

        Args:
            model (ModelDescBase): to be used to obtain inputs_desc and tower_func.
            tower_func: a callable which takes input tensors (by positional args) and construct a tower.
                or a :class:`tfutils.TowerFuncWrapper` instance, which packs both `inputs_desc` and function together.
            inputs_desc ([InputDesc]): if tower_func is a plain function (instead of a TowerFuncWrapper), this describes
                the list of inputs it takes.

            input_names (list): a list of input tensor names. Defaults to match inputs_desc.
                The name can be either the name of a tensor, or the name of one input defined
                by `inputs_desc` or by `model`.
            output_names (list): a list of names of the output tensors to predict, the
                tensors can be any tensor in the graph that's computable from the tensors correponding to `input_names`.

            session_creator (tf.train.SessionCreator): how to create the
                session. Defaults to :class:`tf.train.ChiefSessionCreator()`.
            session_init (SessionInit): how to initialize variables of the session.
                Defaults to do nothing.

            return_input (bool): same as in :attr:`PredictorBase.return_input`.
            create_graph (bool): create a new graph, or use the default graph
                when predictor is first initialized.
        """
        def assert_type(v, tp, name):
            assert isinstance(v, tp), \
                "{} has to be type '{}', but an object of type '{}' found.".format(
                    name, tp.__name__, v.__class__.__name__)

        if model is not None:
            assert_type(model, ModelDescBase, 'model')
            assert inputs_desc is None and tower_func is None
            self.inputs_desc = model.get_inputs_desc()
            self.tower_func = TowerFuncWrapper(model.build_graph, self.inputs_desc)
        else:
            if isinstance(tower_func, TowerFuncWrapper):
                inputs_desc = tower_func.inputs_desc
            assert inputs_desc is not None and tower_func is not None
            self.inputs_desc = inputs_desc
            self.tower_func = TowerFuncWrapper(tower_func, inputs_desc)

        if session_init is None:
            session_init = JustCurrentSession()
        self.session_init = session_init
        assert_type(self.session_init, SessionInit, 'session_init')

        if session_creator is None:
            self.session_creator = tf.train.ChiefSessionCreator(config=get_default_sess_config())
        else:
            self.session_creator = session_creator

        # inputs & outputs
        self.input_names = input_names
        if self.input_names is None:
            self.input_names = [k.name for k in self.inputs_desc]
        self.output_names = output_names
        assert_type(self.output_names, list, 'output_names')
        assert_type(self.input_names, list, 'input_names')
        if len(self.input_names) == 0:
            logger.warn('PredictConfig receives empty "input_names".')
        # assert len(self.input_names), self.input_names
        for v in self.input_names:
            assert_type(v, six.string_types, 'Each item in input_names')
        assert len(self.output_names), self.output_names

        self.return_input = bool(return_input)
        self.create_graph = bool(create_graph)

    def _maybe_create_graph(self):
        if self.create_graph:
            return tf.Graph()
        return tf.get_default_graph()
