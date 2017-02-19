# -*- coding: UTF-8 -*-
# File: trainer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from .base import Trainer

from ..utils import SUMMARY_BACKUP_KEYS, PREDICT_TOWER
from ..tfutils import get_tensors_by_names, TowerContext, get_op_tensor_name
from ..tfutils.collection import freeze_collection
from ..predict import OnlinePredictor, build_prediction_graph
from .input_data import FeedInput

__all__ = ['SimpleTrainer', 'MultiPredictorTowerTrainer']


class PredictorFactory(object):
    """ Make predictors for a trainer"""

    def __init__(self, model, towers):
        """
        :param towers: list of gpu relative id
        """
        self.model = model
        self.towers = towers
        self.tower_built = False

    def get_predictor(self, input_names, output_names, tower):
        """
        Args:
            tower: need the kth tower (not the gpu id)
        Returns:
            an online predictor (which has to be used under a default session)
        """
        if not self.tower_built:
            self._build_predict_tower()
        tower = self.towers[tower % len(self.towers)]

        placeholder_names = set([k.name for k in self.model.get_inputs_desc()])

        def get_name_in_tower(name):
            return PREDICT_TOWER + str(tower) + '/' + name

        def maybe_inside_tower(name):
            name = get_op_tensor_name(name)[0]
            if name in placeholder_names:
                return name
            else:
                return get_name_in_tower(name)

        input_names = map(maybe_inside_tower, input_names)
        raw_input_vars = get_tensors_by_names(input_names)

        output_names = map(get_name_in_tower, output_names)
        output_vars = get_tensors_by_names(output_names)
        return OnlinePredictor(raw_input_vars, output_vars)

    def _build_predict_tower(self):
        # build_predict_tower might get called anywhere, but 'PREDICT_TOWER' should be the outermost name scope
        with tf.name_scope(None), \
                freeze_collection(SUMMARY_BACKUP_KEYS), \
                tf.variable_scope(tf.get_variable_scope(), reuse=True):
            def fn(_):
                self.model.build_graph(self.model.get_reused_placehdrs())
            build_prediction_graph(fn, self.towers)
        self.tower_built = True


class SimpleTrainer(Trainer):
    """ A naive demo trainer which iterates over a DataFlow and feed into the
    graph. It's not efficient compared to QueueInputTrainer or others."""

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): the training config.
        """
        super(SimpleTrainer, self).__init__(config)
        self._predictor_factory = PredictorFactory(self.model, [0])
        if config.dataflow is None:
            self._input_method = config.data
            assert isinstance(self._input_method, FeedInput), type(self._input_method)
        else:
            self._input_method = FeedInput(config.dataflow)

    def run_step(self):
        """ Feed data into the graph and run the updates. """
        feed = self._input_method.next_feed()
        self.monitored_sess.run(self.train_op, feed_dict=feed)

    def _setup(self):
        self._input_method._setup(self)
        model = self.model
        self.input_vars = model.get_reused_placehdrs()
        with TowerContext('', is_training=True):
            model.build_graph(self.input_vars)
            cost_var = model.get_cost()

        opt = self.config.optimizer
        grads = opt.compute_gradients(cost_var)
        self.train_op = opt.apply_gradients(grads, name='min_op')

    def _trigger_epoch(self):
        if self.summary_op is not None:
            feed = self._input_method.last_feed()
            summary_str = self.summary_op.eval(feed_dict=feed)
            self.add_summary(summary_str)

    def get_predict_func(self, input_names, output_names):
        return self._predictor_factory.get_predictor(input_names, output_names, 0)


class MultiPredictorTowerTrainer(Trainer):
    """ A trainer with possibly multiple prediction tower """

    def _setup_predictor_factory(self):
        # by default, use the first training gpu for prediction
        self._predictor_factory = PredictorFactory(
            self.model, self.config.predict_tower)

    def get_predict_func(self, input_names, output_names, tower=0):
        """
        Args:
            tower (int): return the kth predict_func

        Returns:
            an OnlinePredictor instance
        """
        return self._predictor_factory.get_predictor(input_names, output_names, tower)

    def get_predict_funcs(self, input_names, output_names, n):
        return [self.get_predict_func(input_names, output_names, k) for k in range(n)]
