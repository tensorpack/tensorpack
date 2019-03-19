#!/usr/bin/env python

from tensorflow.python.training.monitored_session import _HookedSession as HookedSession

from ..callbacks import Callbacks
from ..tfutils.tower import PredictTowerContext
from .base import PredictorBase

__all__ = ['FeedfreePredictor']


class FeedfreePredictor(PredictorBase):
    """
    Create a predictor that takes inputs from an :class:`InputSource`, instead of from feeds.
    An instance `pred` of :class:`FeedfreePredictor` can be called only by `pred()`, which returns
    a list of output values as defined in config.output_names.
    """

    def __init__(self, config, input_source):
        """
        Args:
            config (PredictConfig): the config to use.
            input_source (InputSource): the feedfree InputSource to use.
                Must match the signature of the tower function in config.
        """
        self._config = config
        self._input_source = input_source
        assert config.return_input is False, \
            "return_input is not supported in FeedfreePredictor! " \
            "If you need to fetch inputs, add the names to the output_names!"

        self._hooks = []
        self.graph = config._maybe_create_graph()
        with self.graph.as_default():
            self._input_callbacks = Callbacks(
                self._input_source.setup(config.input_signature))
            with PredictTowerContext(''):
                self._input_tensors = self._input_source.get_input_tensors()
                config.tower_func(*self._input_tensors)
                self._tower_handle = config.tower_func.towers[-1]

            self._output_tensors = self._tower_handle.get_tensors(config.output_names)

            self._input_callbacks.setup_graph(None)

            for h in self._input_callbacks.get_hooks():
                self._register_hook(h)
            self._initialize_session()

    def _register_hook(self, hook):
        """
        Args:
            hook (tf.train.SessionRunHook):
        """
        self._hooks.append(hook)

    def _initialize_session(self):
        # init the session
        self._config.session_init._setup_graph()
        self._sess = self._config.session_creator.create_session()
        self._config.session_init._run_init(self._sess)

        with self._sess.as_default():
            self._input_callbacks.before_train()
            self._hooked_sess = HookedSession(self._sess, self._hooks)

    def __call__(self):
        return self._hooked_sess.run(self._output_tensors)

    def _do_call(self):
        raise NotImplementedError("You're calling the wrong function!")
