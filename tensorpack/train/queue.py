#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: queue.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import threading
import tensorflow as tf

from ..dataflow.common import RepeatedData
from ..tfutils.summary import summary_moving_average, add_moving_summary
from ..tfutils import get_global_step_var, TowerContext
from ..utils import logger
from ..callbacks.concurrency import StartProcOrThread
from ..tfutils.gradproc import apply_grad_processors
from .inputmethod import QueueInput

from .trainer import (FeedfreeTrainer, MultiPredictorTowerTrainer,
        SingleCostFeedfreeTrainer)

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
        super(QueueInputTrainer, self).__init__(config)
        self._setup_predictor_factory(predict_tower)
        self._input_method = QueueInput(config.dataset, input_queue)

    def _setup(self):
        super(QueueInputTrainer, self)._setup()
        assert len(self.config.tower) == 1, \
                "QueueInputTrainer doesn't support multigpu! Use Sync/AsyncMultiGPUTrainer instead."
        with TowerContext(''):
            cost, grads = self._get_cost_and_grad()
        grads = apply_grad_processors(grads, self.model.get_gradient_processor())

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')
        # skip training
        #self.train_op = tf.group(*self.dequed_inputs)

