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
        When DataFlow is given instead of InputSource, the InputSource to be used will be ``FeedInput(df)``.
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

    @staticmethod
    def setup_graph(model, input):
        """
        Setup graph for simple trainer.

        Args:
            model (ModelDesc):
            input (InputSource):

        Returns:
            tf.Operation: the training op
            [Callback]: the callbacks to be added
        """
        input.setup(model.get_inputs_desc())
        cbs = input.get_callbacks()
        with TowerContext('', is_training=True):
            model.build_graph(input)
            _, grads = model.get_cost_and_grad()
        opt = model.get_optimizer()
        train_op = opt.apply_gradients(grads, name='min_op')
        return train_op, cbs

    def _setup(self):
        self.train_op, callbacks = SimpleTrainer.setup_graph(self.model, self._input_source)
        self.config.callbacks.extend(callbacks)
