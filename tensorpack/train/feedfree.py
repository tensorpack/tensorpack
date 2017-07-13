#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: feedfree.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from six.moves import zip

from ..utils import logger
from ..tfutils.gradproc import FilterNoneGrad
from ..tfutils.tower import TowerContext, get_current_tower_context
from .input_source import QueueInput, FeedfreeInput

from .base import Trainer

__all__ = ['FeedfreeTrainerBase', 'SingleCostFeedfreeTrainer',
           'SimpleFeedfreeTrainer', 'QueueInputTrainer']


class FeedfreeTrainerBase(Trainer):
    """ A base trainer which runs iteration without feed_dict (therefore faster)
        Expect ``self.data`` to be a :class:`FeedfreeInput`.
    """

    # TODO deprecated
    def build_train_tower(self):
        logger.warn("build_train_tower() was deprecated! Please build the graph "
                    "yourself, e.g. by self.model.build_graph(self._input_source)")
        with TowerContext('', is_training=True):
            self.model.build_graph(self._input_source)

    def _setup(self):
        assert isinstance(self._input_source, FeedfreeInput), type(self._input_source)
        self._setup_input_source(self._input_source)

    def run_step(self):
        """ Simply run ``self.train_op``."""
        self.hooked_sess.run(self.train_op)


class SingleCostFeedfreeTrainer(FeedfreeTrainerBase):
    """ A feedfree Trainer which assumes a single cost. """
    def _get_cost_and_grad(self):
        """ get the cost and gradient"""
        ctx = get_current_tower_context()
        assert ctx.is_training, ctx

        self.model.build_graph(self._input_source)
        cost = self.model.get_cost()    # assume single cost

        # produce gradients
        varlist = ctx.filter_vars_by_vs_name(tf.trainable_variables())
        grads = tf.gradients(
            cost,
            varlist,
            gate_gradients=False,
            colocate_gradients_with_ops=True)
        grads = list(zip(grads, varlist))
        grads = FilterNoneGrad().process(grads)
        return cost, grads


class SimpleFeedfreeTrainer(SingleCostFeedfreeTrainer):
    """
    A trainer with single cost, single training tower, any number of
    prediction tower, and feed-free input.
    """

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): ``config.data`` must exist and is a :class:`FeedfreeInput`.
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
        opt = self.model.get_optimizer()
        self.train_op = opt.apply_gradients(grads, name='min_op')


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
    if config.data is not None:
        assert isinstance(config.data, QueueInput), config.data
    else:
        config.data = QueueInput(config.dataflow, input_queue)

    # debug
    # from tensorpack.train.input_source import StagingInputWrapper, DummyConstantInput
    # config.data = StagingInputWrapper(config.data, ['/gpu:0'])
    # config.data = DummyConstantInput([[128,224,224,3], [128]])
    return SimpleFeedfreeTrainer(config)
