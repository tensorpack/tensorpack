# -*- coding: UTF-8 -*-
# File: trainer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import time
from six.moves import zip

from .base import Trainer

from ..dataflow.common import RepeatedData

from ..utils import logger, SUMMARY_BACKUP_KEYS
from ..tfutils import (get_tensors_by_names, freeze_collection,
        get_global_step_var, TowerContext)
from ..tfutils.summary import summary_moving_average, add_moving_summary
from ..predict import OnlinePredictor, build_multi_tower_prediction_graph
from ..tfutils.gradproc import apply_grad_processors

__all__ = ['SimpleTrainer', 'FeedlessTrainer', 'MultiPredictorTowerTrainer',
        'SingleCostFeedlessTrainer']

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
        output_names = ['towerp{}/'.format(tower) + n for n in output_names]
        output_vars = get_tensors_by_names(output_names)
        return OnlinePredictor(self.sess, raw_input_vars, output_vars)

    def _build_predict_tower(self):
        tf.get_variable_scope().reuse_variables()
        # build_predict_tower might get called anywhere, but 'towerp' should be the outermost name scope
        with tf.name_scope(None), \
                freeze_collection(SUMMARY_BACKUP_KEYS):
            build_multi_tower_prediction_graph(self.model, self.towers)
        self.tower_built = True

class SimpleTrainer(Trainer):
    """ A naive demo trainer """
    def __init__(self, config):
        super(SimpleTrainer, self).__init__(config)
        self._predictor_factory = PredictorFactory(self.sess, self.model, [0])

    def run_step(self):
        data = next(self.data_producer)
        feed = dict(zip(self.input_vars, data))
        self.sess.run([self.train_op], feed_dict=feed)    # faster since train_op return None

    def _setup(self):
        model = self.model
        self.input_vars = model.get_input_vars()
        with TowerContext(''):
            model.build_graph(self.input_vars)
            cost_var = model.get_cost()
            add_moving_summary(cost_var)

        grads = self.config.optimizer.compute_gradients(cost_var)
        grads = apply_grad_processors(grads,
                self.model.get_gradient_processor())

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')

        # create an infinte data producer
        self.config.dataset.reset_state()
        self.data_producer = RepeatedData(self.config.dataset, -1).get_data()

    def _trigger_epoch(self):
        if self.summary_op is not None:
            data = next(self.data_producer)
            feed = dict(zip(self.input_vars, data))
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

class FeedlessTrainer(Trainer):
    """ A trainer which runs iteration without feed_dict (therefore faster) """
    def _trigger_epoch(self):
        # need to run summary_op every epoch
        # note that summary_op will take a data from the queue
        if self.summary_op is not None:
            summary_str = self.summary_op.eval()
            self._process_summary(summary_str)

    def _get_input_tensors_noreuse(self):
        """ return a list of actual input tensors.
            Always return new tensors (for multi tower) if called mutliple times.
        """

class SingleCostFeedlessTrainer(Trainer):
    def _get_cost_and_grad(self):
        """ get the cost and gradient on a new tower"""
        actual_inputs = self._get_input_tensors_noreuse()
        self.model.build_graph(actual_inputs)
        cost_var = self.model.get_cost()
        # GATE_NONE faster?
        grads = self.config.optimizer.compute_gradients(
                cost_var, gate_gradients=0)
        add_moving_summary(cost_var)
        return cost_var, grads

