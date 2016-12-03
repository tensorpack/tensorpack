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
from .input_data import QueueInput

from .trainer import (MultiPredictorTowerTrainer, SingleCostFeedfreeTrainer)

__all__ = ['QueueInputTrainer']

class QueueInputTrainer(MultiPredictorTowerTrainer, SingleCostFeedfreeTrainer):
    """ Single GPU Trainer, takes input from a queue"""

    def __init__(self, config, input_queue=None, predict_tower=None):
        """
        :param config: a `TrainConfig` instance
        :param input_queue: a `tf.QueueBase` instance to be used to buffer datapoints.
            Defaults to a FIFO queue of size 100.
        :param predict_tower: list of gpu relative idx to run prediction. default to be [0].
            Use -1 for cpu.
        """
        if hasattr(config, 'dataset'):
            self._input_method = QueueInput(config.dataset, input_queue)
        else:
            self._input_method = config.data
            assert isinstance(self._input_method, QueueInput)
        super(QueueInputTrainer, self).__init__(config)

        self._setup_predictor_factory(predict_tower)
        assert len(self.config.tower) == 1, \
                "QueueInputTrainer doesn't support multigpu! Use Sync/AsyncMultiGPUTrainer instead."

    def _setup(self):
        super(SingleCostFeedfreeTrainer, self)._setup()
        with TowerContext(''):
            cost, grads = self._get_cost_and_grad()
        grads = apply_grad_processors(grads, self.model.get_gradient_processor())

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')
        # skip training
        #self.train_op = tf.group(*self.dequed_inputs)
