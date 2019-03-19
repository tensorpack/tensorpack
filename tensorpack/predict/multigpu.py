# -*- coding: utf-8 -*-
# File: multigpu.py


import tensorflow as tf

from ..input_source import PlaceholderInput
from ..tfutils.tower import PredictTowerContext
from ..utils import logger
from .base import OnlinePredictor

__all__ = ['MultiTowerOfflinePredictor',
           'DataParallelOfflinePredictor']


class MultiTowerOfflinePredictor(OnlinePredictor):
    """ A multi-tower multi-GPU predictor.
        It builds one predictor for each tower.
    """

    def __init__(self, config, towers):
        """
        Args:
            config (PredictConfig): the config to use.
            towers: a list of relative GPU id.
        """
        assert len(towers) > 0
        self.graph = config._maybe_create_graph()
        self.predictors = []
        self.return_input = config.return_input
        with self.graph.as_default():
            handles = []

            input = PlaceholderInput()
            input.setup(config.input_signature)

            for idx, t in enumerate(towers):
                tower_name = 'tower' + str(t)

                device = '/gpu:{}'.format(t)
                with tf.variable_scope(tf.get_variable_scope(), reuse=idx > 0), \
                        tf.device(device), \
                        PredictTowerContext(tower_name):
                    logger.info("Building graph for predict tower '{}' on device {} ...".format(tower_name, device))
                    config.tower_func(*input.get_input_tensors())
                    handles.append(config.tower_func.towers[-1])

            config.session_init._setup_graph()
            self.sess = config.session_creator.create_session()
            config.session_init._run_init(self.sess)

            for h in handles:
                input_tensors = h.get_tensors(config.input_names)
                output_tensors = h.get_tensors(config.output_names)
                self.predictors.append(OnlinePredictor(
                    input_tensors, output_tensors, config.return_input, self.sess))

    def _do_call(self, dp):
        # use the first tower for compatible PredictorBase interface
        return self.predictors[0]._do_call(dp)

    def get_predictor(self, n):
        """
        Returns:
            OnlinePredictor: the nth predictor on the nth tower.
        """
        l = len(self.predictors)
        if n >= l:
            logger.warn("n > #towers, will assign predictor to GPU by round-robin")
        return [self.predictors[k % l] for k in range(n)]

    def get_predictors(self):
        """
        Returns:
            list[OnlinePredictor]: a list of predictor
        """
        return self.predictors


class DataParallelOfflinePredictor(OnlinePredictor):
    """
    A data-parallel predictor. It builds one predictor that utilizes all GPUs.

    Note that it doesn't split/concat inputs/outputs automatically.
    Instead, its inputs are:
    ``[input[0] in tower[0], input[1] in tower[0], ..., input[0] in tower[1], input[1] in tower[1], ...]``
    Similar for the outputs.
    """

    def __init__(self, config, towers):
        """
        Args:
            config (PredictConfig): the config to use.
            towers: a list of relative GPU id.
        """
        self.graph = config._maybe_create_graph()
        with self.graph.as_default():
            input_tensors = []
            output_tensors = []

            for idx, t in enumerate(towers):
                tower_name = 'tower' + str(t)

                new_sig = [tf.TensorSpec(dtype=p.dtype, shape=p.shape, name=tower_name + '_' + p.name)
                           for p in config.input_signature]
                input = PlaceholderInput()
                input.setup(new_sig)

                with tf.variable_scope(tf.get_variable_scope(), reuse=idx > 0), \
                        tf.device('/gpu:{}'.format(t)), \
                        PredictTowerContext(tower_name):
                    config.tower_func(*input.get_input_tensors())
                    h = config.tower_func.towers[-1]
                    input_tensors.extend(h.get_tensors(config.input_names))
                    output_tensors.extend(h.get_tensors(config.output_names))

            config.session_init._setup_graph()
            sess = config.session_creator.create_session()
            config.session_init._run_init(sess)
            super(DataParallelOfflinePredictor, self).__init__(
                input_tensors, output_tensors, config.return_input, sess)
