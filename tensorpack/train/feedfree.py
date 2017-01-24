#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: feedfree.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..utils import logger
from ..tfutils.tower import TowerContext
from ..tfutils.gradproc import apply_grad_processors
from .input_data import QueueInput, FeedfreeInput

from .base import Trainer
from .trainer import MultiPredictorTowerTrainer

__all__ = ['FeedfreeTrainerBase', 'SingleCostFeedfreeTrainer',
           'SimpleFeedfreeTrainer', 'QueueInputTrainer']


class FeedfreeTrainerBase(Trainer):
    """ A base trainer which runs iteration without feed_dict (therefore faster)
        Expect ``self.data`` to be a :class:`FeedfreeInput`.
    """

    def _trigger_epoch(self):
        # run summary_op every epoch
        # TODO FIXME summary_op will take a data! This is not good for TensorInput.
        if self.summary_op is not None:
            summary_str = self.summary_op.eval()
            self.add_summary(summary_str)

    def _get_input_tensors(self):
        return self._input_method.get_input_tensors()

    def _setup(self):
        assert isinstance(self._input_method, FeedfreeInput), type(self._input_method)
        self._input_method._setup(self)


class SingleCostFeedfreeTrainer(FeedfreeTrainerBase):
    """ A feedfree Trainer which assumes a single cost. """
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
        return cost_var, grads

    def run_step(self):
        """ Simply run ``self.train_op``, which minimizes the cost."""
        ret = self.sess.run([self.train_op] + self.get_extra_fetches())
        return ret[1:]
        # if not hasattr(self, 'cnt'):
        #     self.cnt = 0
        # else:
        #     self.cnt += 1
        #     if self.cnt % 10 == 0:
        #     # debug-benchmark code:
        #         run_metadata = tf.RunMetadata()
        #         self.sess.run([self.train_op],
        #                 options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #                 run_metadata=run_metadata
        #                 )
        #         from tensorflow.python.client import timeline
        #         trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        #         trace_file = open('timeline.ctf.json', 'w')
        #         trace_file.write(trace.generate_chrome_trace_format())
        #         import sys; sys.exit()


class SimpleFeedfreeTrainer(
        SingleCostFeedfreeTrainer,
        MultiPredictorTowerTrainer):
    """
    A trainer with single cost, single training tower, any number of
    prediction tower, and feed-free input.
    """

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): ``config.data`` must exist and is a
                :class:`FeedfreeInput`.
        """
        self._input_method = config.data
        assert isinstance(self._input_method, FeedfreeInput), self._input_method
        super(SimpleFeedfreeTrainer, self).__init__(config)
        self._setup_predictor_factory()
        assert len(self.config.tower) == 1, \
            "SimpleFeedfreeTrainer doesn't support multigpu!"

    def _setup(self):
        super(SimpleFeedfreeTrainer, self)._setup()
        with TowerContext('', is_training=True):
            cost, grads = self._get_cost_and_grad()
        grads = apply_grad_processors(grads, self.model.get_gradient_processor())

        self.train_op = self.config.optimizer.apply_gradients(grads, name='min_op')
        # skip training
        # self.train_op = tf.group(*self.dequed_inputs)


class QueueInputTrainer(SimpleFeedfreeTrainer):
    """
    A trainer which automatically wraps ``config.dataflow`` by a
    :class:`QueueInput`.
    """

    def __init__(self, config, input_queue=None, predict_tower=None):
        """
        Single tower Trainer, takes input from a queue

        Args:
            config(TrainConfig): a `TrainConfig` instance. config.dataflow must exist.
            input_queue(tf.QueueBase): an input queue. Defaults to the
                :class:`QueueInput` default.
        """
        config.data = QueueInput(config.dataflow, input_queue)
        if predict_tower is not None:
            logger.warn("[Deprecated] Argument `predict_tower` is deprecated for trainer. "
                        "Use TrainConfig(predict_tower=...) instead!")
            config.predict_tower = predict_tower
        assert len(config.tower) == 1, \
            "QueueInputTrainer doesn't support multigpu! Use Sync/AsyncMultiGPUTrainer instead."
        super(QueueInputTrainer, self).__init__(config)
