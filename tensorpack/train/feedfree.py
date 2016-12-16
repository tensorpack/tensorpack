#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: feedfree.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..utils import logger
from ..tfutils import get_global_step_var
from ..tfutils.tower import TowerContext
from ..tfutils.gradproc import apply_grad_processors
from ..tfutils.summary import summary_moving_average, add_moving_summary
from .input_data import QueueInput, FeedfreeInput, DummyConstantInput

from .base import Trainer
from .trainer import MultiPredictorTowerTrainer

__all__ = ['FeedfreeTrainer', 'SingleCostFeedfreeTrainer', 'SimpleFeedfreeTrainer', 'QueueInputTrainer']

class FeedfreeTrainer(Trainer):
    """ A trainer which runs iteration without feed_dict (therefore faster) """
    def _trigger_epoch(self):
        # need to run summary_op every epoch
        # note that summary_op will take a data from the queue
        if self.summary_op is not None:
            summary_str = self.summary_op.eval()
            self._process_summary(summary_str)

    def _get_input_tensors(self):
        return self._input_method.get_input_tensors()

    def _setup(self):
        assert isinstance(self._input_method, FeedfreeInput), type(self._input_method)
        self._input_method._setup(self)

class SingleCostFeedfreeTrainer(FeedfreeTrainer):
    def _get_cost_and_grad(self):
        """ get the cost and gradient on a new tower"""
        actual_inputs = self._get_input_tensors()
        self.model.build_graph(actual_inputs)
        cost_var = self.model.get_cost()
        # GATE_NONE faster?
        grads = self.config.optimizer.compute_gradients(
                cost_var,
                gate_gradients=tf.train.Optimizer.GATE_NONE,
                colocate_gradients_with_ops=False)
        add_moving_summary(cost_var)
        return cost_var, grads

    def run_step(self):
        """ Simply run self.train_op"""
        self.sess.run(self.train_op)
        # debug-benchmark code:
        #run_metadata = tf.RunMetadata()
        #self.sess.run([self.train_op],
                #options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                #run_metadata=run_metadata
                #)
        #from tensorflow.python.client import timeline
        #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        #trace_file = open('timeline.ctf.json', 'w')
        #trace_file.write(trace.generate_chrome_trace_format())
        #import sys; sys.exit()

class SimpleFeedfreeTrainer(
        MultiPredictorTowerTrainer,
        SingleCostFeedfreeTrainer):
    def __init__(self, config):
        """
        A trainer with single cost, single training tower and feed-free input
        config.data must exists
        """
        self._input_method = config.data
        assert isinstance(self._input_method, FeedfreeInput), self._input_method
        super(SimpleFeedfreeTrainer, self).__init__(config)
        self._setup_predictor_factory(config.predict_tower)
        assert len(self.config.tower) == 1, \
                "SimpleFeedfreeTrainer doesn't support multigpu!"

    def _setup(self):
        super(SimpleFeedfreeTrainer, self)._setup()
        with TowerContext('', is_training=True):
            cost, grads = self._get_cost_and_grad()
        grads = apply_grad_processors(grads, self.model.get_gradient_processor())

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')
        # skip training
        #self.train_op = tf.group(*self.dequed_inputs)

class QueueInputTrainer(SimpleFeedfreeTrainer):

    def __init__(self, config, input_queue=None, predict_tower=None):
        """
        Single tower Trainer, takes input from a queue

        :param config: a `TrainConfig` instance. config.dataset must exist
        :param input_queue: a `tf.QueueBase` instance
        :param predict_tower: list of gpu relative idx to run prediction. default to be [0].
            Use -1 for cpu.
        """
        config.data = QueueInput(config.dataset, input_queue)
        if predict_tower is not None:
            logger.warn("[Deprecated] Argument `predict_tower` is deprecated for trainer. Use TrainConfig.predict_tower instead!")
            config.predict_tower = predict_tower
        assert len(config.tower) == 1, \
                "QueueInputTrainer doesn't support multigpu! Use Sync/AsyncMultiGPUTrainer instead."
        super(QueueInputTrainer, self).__init__(config)
