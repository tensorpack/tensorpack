#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: queue.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..utils import logger
from ..tfutils import get_global_step_var
from ..tfutils.tower import TowerContext
from ..tfutils.gradproc import apply_grad_processors
from ..tfutils.summary import summary_moving_average
from .input_data import QueueInput, FeedfreeInput

from .trainer import (MultiPredictorTowerTrainer, SingleCostFeedfreeTrainer)

__all__ = ['SimpleFeedfreeTrainer', 'QueueInputTrainer']

class SimpleFeedfreeTrainer(
        MultiPredictorTowerTrainer,
        SingleCostFeedfreeTrainer):
    def __init__(self, config, predict_tower=None):
        """
        A trainer with single cost, single training tower and feed-free input
        config.data must exists
        """
        self._input_method = config.data
        assert isinstance(self._input_method, FeedfreeInput), self._input_method
        super(SimpleFeedfreeTrainer, self).__init__(config)
        self._setup_predictor_factory(predict_tower)
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
        :param input_queue: a `tf.QueueBase` instance to be used to buffer datapoints.
            Defaults to a FIFO queue of size 100.
        :param predict_tower: list of gpu relative idx to run prediction. default to be [0].
            Use -1 for cpu.
        """
        config.data = QueueInput(config.dataset, input_queue)
        assert len(config.tower) == 1, \
                "QueueInputTrainer doesn't support multigpu! Use Sync/AsyncMultiGPUTrainer instead."
        super(QueueInputTrainer, self).__init__(config, predict_tower)

    def _setup(self):
        super(QueueInputTrainer, self)._setup()
