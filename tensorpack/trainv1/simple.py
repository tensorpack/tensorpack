# -*- coding: UTF-8 -*-
# File: simple.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


from .base import Trainer

from ..tfutils.tower import TowerContext
from ..utils import logger
from ..input_source import FeedInput, QueueInput

__all__ = ['SimpleTrainer', 'QueueInputTrainer']


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

    def _setup(self):
        cbs = self._input_source.setup(self.model.get_inputs_desc())

        with TowerContext('', is_training=True):
            grads = self.model._build_graph_get_grads(
                *self._input_source.get_input_tensors())
            opt = self.model.get_optimizer()
            self.train_op = opt.apply_gradients(grads, name='min_op')

        self._config.callbacks.extend(cbs)


def QueueInputTrainer(config, input_queue=None):
    """
    A wrapper trainer which automatically wraps ``config.dataflow`` by a :class:`QueueInput`.
    It is an equivalent of ``SimpleTrainer(config)`` with ``config.data = QueueInput(dataflow)``.

    Args:
        config (TrainConfig): Must contain 'model' and 'dataflow'.
        input_queue (tf.QueueBase): an input queue. Defaults to the :class:`QueueInput` default.
    """
    assert (config.data is not None or config.dataflow is not None) and config.model is not None
    if config.data is not None:
        assert isinstance(config.data, QueueInput), config.data
    else:
        config.data = QueueInput(config.dataflow, input_queue)
    config.dataflow = None
    return SimpleTrainer(config)
