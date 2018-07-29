# -*- coding: utf-8 -*-
# File: interface.py

import tensorflow as tf

from ..input_source import (
    InputSource, FeedInput, FeedfreeInput,
    QueueInput, StagingInput, DummyConstantInput)
from ..utils import logger

from .config import TrainConfig
from .tower import SingleCostTrainer
from .trainers import SimpleTrainer

__all__ = ['launch_train_with_config', 'apply_default_prefetch']


def apply_default_prefetch(input_source_or_dataflow, trainer):
    """
    Apply a set of default rules to make a fast :class:`InputSource`.

    Args:
        input_source_or_dataflow(InputSource | DataFlow):
        trainer (Trainer):

    Returns:
        InputSource
    """
    if not isinstance(input_source_or_dataflow, InputSource):
        # to mimic same behavior of the old trainer interface
        if type(trainer) == SimpleTrainer:
            input = FeedInput(input_source_or_dataflow)
        else:
            logger.info("Automatically applying QueueInput on the DataFlow.")
            input = QueueInput(input_source_or_dataflow)
    else:
        input = input_source_or_dataflow
    if hasattr(trainer, 'devices'):
        towers = trainer.devices
        if len(towers) > 1:
            # seem to only improve on >1 GPUs
            assert not isinstance(trainer, SimpleTrainer)

            if isinstance(input, FeedfreeInput) and \
               not isinstance(input, (StagingInput, DummyConstantInput)):
                logger.info("Automatically applying StagingInput on the DataFlow.")
                input = StagingInput(input)
    return input


def launch_train_with_config(config, trainer):
    """
    Train with a :class:`TrainConfig` and a :class:`Trainer`, to
    present a simple training interface. It basically does the following
    3 things (and you can easily do them by yourself if you need more control):

    1. Setup the input with automatic prefetching heuristics,
       from `config.data` or `config.dataflow`.
    2. Call `trainer.setup_graph` with the input as well as `config.model`.
    3. Call `trainer.train` with rest of the attributes of config.

    Args:
        config (TrainConfig):
        trainer (Trainer): an instance of :class:`SingleCostTrainer`.

    Example:

    .. code-block:: python

        launch_train_with_config(
            config, SyncMultiGPUTrainerParameterServer(8, ps_device='gpu'))
    """
    assert isinstance(trainer, SingleCostTrainer), trainer
    assert isinstance(config, TrainConfig), config
    assert config.model is not None
    assert config.dataflow is not None or config.data is not None

    model = config.model
    inputs_desc = model.get_inputs_desc()
    input = config.data or config.dataflow
    input = apply_default_prefetch(input, trainer)

    trainer.setup_graph(
        inputs_desc, input,
        model._build_graph_get_cost, model.get_optimizer)
    _check_unused_regularization()
    trainer.train_with_defaults(
        callbacks=config.callbacks,
        monitors=config.monitors,
        session_creator=config.session_creator,
        session_init=config.session_init,
        steps_per_epoch=config.steps_per_epoch,
        starting_epoch=config.starting_epoch,
        max_epoch=config.max_epoch,
        extra_callbacks=config.extra_callbacks)


def _check_unused_regularization():
    coll = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    unconsumed_reg = []
    for c in coll:
        if len(c.consumers()) == 0:
            unconsumed_reg.append(c)
    if unconsumed_reg:
        logger.warn("The following tensors appear in REGULARIZATION_LOSSES collection but have no "
                    "consumers! You may have forgotten to add regularization to total cost.")
        logger.warn("Unconsumed regularization: {}".format(', '.join([x.name for x in unconsumed_reg])))
