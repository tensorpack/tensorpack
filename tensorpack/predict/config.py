# -*- coding: utf-8 -*-
# File: config.py


import six
from ..compat import tfv1 as tf

from ..train.model_desc import ModelDescBase
from ..tfutils.sessinit import JustCurrentSession, SessionInit
from ..tfutils.sesscreate import NewSessionCreator
from ..tfutils.tower import TowerFunc
from ..utils import logger
from ..utils.develop import log_deprecated

__all__ = ['PredictConfig']


class PredictConfig(object):
    def __init__(self,
                 model=None,
                 tower_func=None,
                 input_signature=None,

                 input_names=None,
                 output_names=None,

                 session_creator=None,
                 session_init=None,
                 return_input=False,
                 create_graph=True,
                 inputs_desc=None
                 ):
        """
        Users need to provide enough arguments to create a tower function,
        which will be used to construct the graph.
        This can be provided in the following ways:

        1. `model`: a :class:`ModelDesc` instance. It will contain a tower function by itself.
        2. `tower_func`: a :class:`tfutils.TowerFunc` instance.
            Provide a tower function instance directly.
        3. `tower_func`: a symbolic function and `input_signature`: the signature of the function.
            Provide both a function and its signature.

        Example:

        .. code-block:: python

            config = PredictConfig(model=my_model,
                                   inputs_names=['image'],
                                   output_names=['linear/output', 'prediction'])

        Args:
            model (ModelDescBase): to be used to construct a tower function.
            tower_func: a callable which takes input tensors (by positional args) and construct a tower.
                or a :class:`tfutils.TowerFunc` instance.
            input_signature ([tf.TensorSpec]): if tower_func is a plain function (instead of a TowerFunc),
                this describes the list of inputs it takes.

            input_names (list): a list of input tensor names. Defaults to match input_signature.
                The name can be either the name of a tensor, or the name of one input of the tower.
            output_names (list): a list of names of the output tensors to predict, the
                tensors can be any tensor in the graph that's computable from the tensors correponding to `input_names`.

            session_creator (tf.train.SessionCreator): how to create the
                session. Defaults to :class:`NewSessionCreator()`.
            session_init (SessionInit): how to initialize variables of the session.
                Defaults to do nothing.

            return_input (bool): same as in :attr:`PredictorBase.return_input`.
            create_graph (bool): create a new graph, or use the default graph
                when predictor is first initialized.
            inputs_desc (list[tf.TensorSpec]): old (deprecated) name for `input_signature`.
        """
        def assert_type(v, tp, name):
            assert isinstance(v, tp), \
                "Argument '{}' has to be type '{}', but an object of type '{}' found.".format(
                    name, tp.__name__, v.__class__.__name__)

        if inputs_desc is not None:
            log_deprecated("PredictConfig(inputs_desc)", "Use input_signature instead!", "2020-03-01")
            assert input_signature is None, "Cannot set both inputs_desc and input_signature!"
            input_signature = inputs_desc

        if model is not None:
            assert_type(model, ModelDescBase, 'model')
            assert input_signature is None and tower_func is None
            self.input_signature = model.get_input_signature()
            self.tower_func = TowerFunc(model.build_graph, self.input_signature)
        else:
            if isinstance(tower_func, TowerFunc):
                input_signature = tower_func.input_signature
            assert input_signature is not None and tower_func is not None
            self.input_signature = input_signature
            self.tower_func = TowerFunc(tower_func, input_signature)

        if session_init is None:
            session_init = JustCurrentSession()
        self.session_init = session_init
        assert_type(self.session_init, SessionInit, 'session_init')

        if session_creator is None:
            self.session_creator = NewSessionCreator()
        else:
            self.session_creator = session_creator

        # inputs & outputs
        self.input_names = input_names
        if self.input_names is None:
            self.input_names = [k.name for k in self.input_signature]
        assert output_names is not None, "Argument 'output_names' is not provided!"
        self.output_names = output_names
        assert_type(self.output_names, list, 'output_names')
        assert_type(self.input_names, list, 'input_names')
        if len(self.input_names) == 0:
            logger.warn('PredictConfig receives empty "input_names".')
        for v in self.input_names:
            assert_type(v, six.string_types, 'Each item in input_names')
        assert len(self.output_names), "Argument 'output_names' cannot be empty!"

        self.return_input = bool(return_input)
        self.create_graph = bool(create_graph)

        self.inputs_desc = input_signature  # TODO a little bit of compatibility

    def _maybe_create_graph(self):
        if self.create_graph:
            return tf.Graph()
        return tf.get_default_graph()
