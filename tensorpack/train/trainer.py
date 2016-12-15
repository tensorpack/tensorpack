# -*- coding: UTF-8 -*-
# File: trainer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import time
from six.moves import zip

from .base import Trainer

from ..utils import logger, SUMMARY_BACKUP_KEYS, PREDICT_TOWER
from ..tfutils import (get_tensors_by_names, freeze_collection,
        get_global_step_var, TowerContext)
from ..tfutils.summary import summary_moving_average, add_moving_summary
from ..predict import OnlinePredictor, build_multi_tower_prediction_graph
from ..tfutils.gradproc import apply_grad_processors
from .input_data import FeedInput, FeedfreeInput

__all__ = ['SimpleTrainer','MultiPredictorTowerTrainer']

class PredictorFactory(object):
    """ Make predictors for a trainer"""

    def __init__(self, sess, model, towers):
        """
        :param towers: list of gpu relative id
        """
        self.sess = sess
        self.model = model
        self.towers = towers
        self.tower_built = False

    def get_predictor(self, input_names, output_names, tower):
        """
        :param tower: need the kth tower (not the gpu id)
        :returns: an online predictor
        """
        if not self.tower_built:
            self._build_predict_tower()
        tower = self.towers[tower % len(self.towers)]
        raw_input_vars = get_tensors_by_names(input_names)
        output_names = ['{}{}/'.format(PREDICT_TOWER, tower) + n for n in output_names]
        output_vars = get_tensors_by_names(output_names)
        return OnlinePredictor(self.sess, raw_input_vars, output_vars)

    def _build_predict_tower(self):
        tf.get_variable_scope().reuse_variables()
        # build_predict_tower might get called anywhere, but 'PREDICT_TOWER' should be the outermost name scope
        with tf.name_scope(None), \
                freeze_collection(SUMMARY_BACKUP_KEYS):
            fn = lambda _: self.model.build_graph(self.model.get_input_vars())
            build_multi_tower_prediction_graph(fn, self.towers)
        self.tower_built = True

class SimpleTrainer(Trainer):
    """ A naive demo trainer """
    def __init__(self, config):
        super(SimpleTrainer, self).__init__(config)
        self._predictor_factory = PredictorFactory(self.sess, self.model, [0])
        if not hasattr(config, 'dataset'):
            self._input_method = config.data
            assert isinstance(self._input_method, FeedInput)
        else:
            self._input_method = FeedInput(config.dataset)

    def run_step(self):
        feed = self._input_method.next_feed()
        self.sess.run([self.train_op], feed_dict=feed)    # faster since train_op return None

    def _setup(self):
        self._input_method._setup(self)
        model = self.model
        self.input_vars = model.get_input_vars()
        with TowerContext('', is_training=True):
            model.build_graph(self.input_vars)
            cost_var = model.get_cost()
            add_moving_summary(cost_var)

        grads = self.config.optimizer.compute_gradients(cost_var)
        grads = apply_grad_processors(grads,
                self.model.get_gradient_processor())

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')

    def _trigger_epoch(self):
        if self.summary_op is not None:
            feed = self._input_method.next_feed()
            summary_str = self.summary_op.eval(feed_dict=feed)
            self._process_summary(summary_str)

    def get_predict_func(self, input_names, output_names):
        return self._predictor_factory.get_predictor(input_names, output_names, 0)

class MultiPredictorTowerTrainer(Trainer):
    """ A trainer with possibly multiple prediction tower """
    def _setup_predictor_factory(self, predict_tower):
        # by default, use the first training gpu for prediction
        predict_tower = predict_tower or [0]
        self._predictor_factory = PredictorFactory(
                self.sess, self.model, predict_tower)

    def get_predict_func(self, input_names, output_names, tower=0):
        """
        :param tower: return the kth predict_func
        :returns: an `OnlinePredictor`
        """
        return self._predictor_factory.get_predictor(input_names, output_names, tower)

    def get_predict_funcs(self, input_names, output_names, n):
        return [self.get_predict_func(input_names, output_names, k) for k in range(n)]
