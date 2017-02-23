# -*- coding: UTF-8 -*-
# File: trainer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from .base import Trainer

from ..tfutils import TowerContext
from .input_data import FeedInput

__all__ = ['SimpleTrainer']


class SimpleTrainer(Trainer):
    """ A naive demo trainer which iterates over a DataFlow and feed into the
    graph. It's not efficient compared to QueueInputTrainer or others."""

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): the training config.
        """
        super(SimpleTrainer, self).__init__(config)
        if config.dataflow is None:
            self._input_method = config.data
            assert isinstance(self._input_method, FeedInput), type(self._input_method)
        else:
            self._input_method = FeedInput(config.dataflow)

    def run_step(self):
        """ Feed data into the graph and run the updates. """
        dp = self._input_method.next_feed()
        feed = dict(zip(self.inputs, dp))
        self.hooked_sess.run(self.train_op, feed_dict=feed)

    def _setup(self):
        self._input_method.setup_training(self)
        model = self.model
        self.inputs = model.get_reused_placehdrs()
        with TowerContext('', is_training=True):
            model.build_graph(self.inputs)
            cost_var = model.get_cost()

        opt = self.config.optimizer
        self.train_op = opt.minimize(cost_var, name='min_op')
