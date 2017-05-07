#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: feedfree.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..tfutils.tower import TowerContext, get_current_tower_context
from .input_source import QueueInput, FeedfreeInput

from .base import Trainer

__all__ = ['FeedfreeTrainerBase', 'SingleCostFeedfreeTrainer',
           'SimpleFeedfreeTrainer', 'QueueInputTrainer']


class FeedfreeTrainerBase(Trainer):
    """ A base trainer which runs iteration without feed_dict (therefore faster)
        Expect ``self.data`` to be a :class:`FeedfreeInput`.
    """
    def build_train_tower(self):
        """
        Get input tensors from `self.input_source` and build the forward graph.
        """
        def f():
            self._input_tensors = self._input_source.get_input_tensors()
            self.model.build_graph(self._input_tensors)
        ctx = get_current_tower_context()
        if ctx is None:     # call without a context, use a default one
            with TowerContext('', is_training=True):
                f()
        else:
            assert ctx.is_training, ctx
            f()

    def _setup(self):
        assert isinstance(self._input_source, FeedfreeInput), type(self._input_source)
        self._input_source.setup_training(self)

    def run_step(self):
        """ Simply run ``self.train_op``."""
        self.hooked_sess.run(self.train_op)
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


class SingleCostFeedfreeTrainer(FeedfreeTrainerBase):
    """ A feedfree Trainer which assumes a single cost. """
    def _get_cost_and_grad(self):
        """ get the cost and gradient"""
        self.build_train_tower()
        cost = self.model.get_cost()    # assume single cost
        opt = self.config.optimizer
        # GATE_NONE faster?
        grads = opt.compute_gradients(
            cost,
            gate_gradients=tf.train.Optimizer.GATE_NONE,
            colocate_gradients_with_ops=True)
        return cost, grads


class SimpleFeedfreeTrainer(SingleCostFeedfreeTrainer):
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
        self._input_source = config.data
        assert isinstance(self._input_source, FeedfreeInput), self._input_source
        super(SimpleFeedfreeTrainer, self).__init__(config)
        assert len(self.config.tower) == 1, \
            "Got nr_tower={}, but doesn't support multigpu!" \
            " Use Sync/AsyncMultiGPUTrainer instead.".format(len(self.config.tower))

    def _setup(self):
        super(SimpleFeedfreeTrainer, self)._setup()
        with TowerContext('', is_training=True):
            cost, grads = self._get_cost_and_grad()
        self.train_op = self.config.optimizer.apply_gradients(grads, name='min_op')
        # skip training
        # self.train_op = tf.group(*self._input_tensors)


def QueueInputTrainer(config, input_queue=None):
    """
    A wrapper trainer which automatically wraps ``config.dataflow`` by a
    :class:`QueueInput`.
    It is an equivalent of ``SimpleFeedfreeTrainer(config)`` with ``config.data = QueueInput(dataflow)``.

    Args:
        config (TrainConfig): a `TrainConfig` instance. config.dataflow must exist.
        input_queue (tf.QueueBase): an input queue. Defaults to the
            :class:`QueueInput` default.
    """
    if config.dataflow is not None:
        config.data = QueueInput(config.dataflow, input_queue)
    else:
        assert isinstance(config.data, QueueInput), config.data

    # debug
    # from tensorpack.train.input_source import StagingInputWrapper, DummyConstantInput
    # config.data = StagingInputWrapper(config.data, ['/gpu:0'])
    # config.data = DummyConstantInput([[128,224,224,3], [128]])
    return SimpleFeedfreeTrainer(config)
