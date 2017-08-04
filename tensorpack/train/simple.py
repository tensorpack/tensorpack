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
        It simply builds one tower and minimize `model.cost`.
        It supports both InputSource and DataFlow.

        When DataFlow is given instead of InputSource, the InputSource to be
        used will be ``FeedInput(df)`` (no prefetch).
    """

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): Must contain 'model' and either one of 'data' or 'dataflow'.
        """
        assert len(config.tower) == 1, \
            "Got nr_tower={}, but doesn't support multigpu!" \
            " Use Sync/AsyncMultiGPUTrainer instead.".format(len(config.tower))

        assert (config.data is not None or config.dataflow is not None) and config.model is not None
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
        Setup graph for SimpleTrainer. It simply build one tower and optimize `model.cost`.

        Args:
            model (ModelDesc):
            input (InputSource):

        Returns:
            tf.Operation: the training op

            [Callback]: the callbacks to be added
        """
        cbs = input.setup(model.get_inputs_desc())
        with TowerContext('', is_training=True):
            model.build_graph(input)
            _, grads = model.get_cost_and_grad()
        opt = model.get_optimizer()
        train_op = opt.apply_gradients(grads, name='min_op')
        return train_op, cbs

    def _setup(self):
        self.train_op, callbacks = SimpleTrainer.setup_graph(self.model, self._input_source)
        self.config.callbacks.extend(callbacks)
