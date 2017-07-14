#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: feedfree.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from ..utils import logger
from ..utils.develop import deprecated
from ..tfutils.tower import TowerContext
from ..graph_builder.input_source import QueueInput, FeedfreeInput

from .simple import SimpleTrainer
from .base import Trainer

__all__ = ['FeedfreeTrainerBase', 'SingleCostFeedfreeTrainer',
           'SimpleFeedfreeTrainer', 'QueueInputTrainer']


class FeedfreeTrainerBase(Trainer):
    """ A base trainer which runs iteration without feed_dict (therefore faster)
        Expect ``config.data`` to be a :class:`FeedfreeInput`.
    """

    @deprecated("Please build the graph yourself, e.g. by self.model.build_graph(self._input_source)")
    def build_train_tower(self):
        with TowerContext('', is_training=True):
            self.model.build_graph(self._input_source)

    def _setup(self):
        assert isinstance(self._input_source, FeedfreeInput), type(self._input_source)
        self._setup_input_source(self._input_source)

    def run_step(self):
        """ Simply run ``self.train_op``."""
        self.hooked_sess.run(self.train_op)


# deprecated
class SingleCostFeedfreeTrainer(FeedfreeTrainerBase):
    """ A feedfree Trainer which assumes a single cost. """
    def __init__(self, *args, **kwargs):
        super(SingleCostFeedfreeTrainer, self).__init__(*args, **kwargs)
        logger.warn("SingleCostFeedfreeTrainer was deprecated!")

    def _get_cost_and_grad(self):
        """ get the cost and gradient"""
        self.model.build_graph(self._input_source)
        return self.model.get_cost_and_grad()


@deprecated("Use SimpleTrainer with config.data is the same!", "2017-09-13")
def SimpleFeedfreeTrainer(config):
    assert isinstance(config.data, FeedfreeInput), config.data
    return SimpleTrainer(config)


def QueueInputTrainer(config, input_queue=None):
    """
    A wrapper trainer which automatically wraps ``config.dataflow`` by a :class:`QueueInput`.
    It is an equivalent of ``SimpleTrainer(config)`` with ``config.data = QueueInput(dataflow)``.

    Args:
        config (TrainConfig): a `TrainConfig` instance. config.dataflow must exist.
        input_queue (tf.QueueBase): an input queue. Defaults to the :class:`QueueInput` default.
    """
    if config.data is not None:
        assert isinstance(config.data, QueueInput), config.data
    else:
        config.data = QueueInput(config.dataflow, input_queue)
    config.dataflow = None

    # debug
    # from tensorpack.train.input_source import StagingInputWrapper, DummyConstantInput
    # config.data = StagingInputWrapper(config.data, ['/gpu:0'])
    # config.data = DummyConstantInput([[128,224,224,3], [128]])
    return SimpleTrainer(config)
