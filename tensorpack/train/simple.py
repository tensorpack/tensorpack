# -*- coding: UTF-8 -*-
# File: simple.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


from .base import Trainer

from ..utils import logger
from ..tfutils import TowerContext
from ..graph_builder.input_source import FeedInput

__all__ = ['SimpleTrainer']


class SimpleTrainer(Trainer):
    """ A naive single-tower single-cost demo trainer.
        Support both InputSource and DataFlow.
        When DataFlow is given, the InputSource to be used will be ``FeedInput(df)``.
    """

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): the training config.
        """
        assert len(config.tower) == 1, \
            "Got nr_tower={}, but doesn't support multigpu!" \
            " Use Sync/AsyncMultiGPUTrainer instead.".format(len(config.tower))

        if config.dataflow is None:
            self._input_source = config.data
        else:
            self._input_source = FeedInput(config.dataflow)
            logger.warn("FeedInput is slow (and this is the default of SimpleTrainer). "
                        "Consider QueueInput or other InputSource instead.")
        super(SimpleTrainer, self).__init__(config)

    def run_step(self):
        self.hooked_sess.run(self.train_op)

    def _setup(self):
        self._setup_input_source(self._input_source)
        with TowerContext('', is_training=True):
            self.model.build_graph(self._input_source)
            cost, grads = self.model.get_cost_and_grad()
        opt = self.model.get_optimizer()
        self.train_op = opt.apply_gradients(grads, name='min_op')
