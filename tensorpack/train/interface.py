#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: interface.py

import tensorflow as tf

from ..input_source import (
    InputSource, FeedInput, QueueInput, StagingInput, DummyConstantInput)

from ..trainv1.config import TrainConfig
from .tower import SingleCostTrainer
from .trainers import SimpleTrainer, DistributedTrainerReplicated

__all__ = ['launch_train_with_config', 'apply_default_prefetch']


def apply_default_prefetch(input_source_or_dataflow, trainer, towers):
    """
    Apply a set of default rules to make a fast :class:`InputSource`.

    Args:
        input_source_or_dataflow(InputSource | DataFlow):
        trainer (Trainer):
        towers ([int]): list of GPU ids.
    """
    if not isinstance(input_source_or_dataflow, InputSource):
        # to mimic same behavior of the old trainer interface
        if type(trainer) == SimpleTrainer:
            input = FeedInput(input_source_or_dataflow)
        else:
            input = QueueInput(input_source_or_dataflow)
    else:
        input = input_source_or_dataflow
    if len(towers) > 1:
        # seem to only improve on >1 GPUs
        assert not isinstance(trainer, SimpleTrainer)
        assert tf.test.is_gpu_available()

        if not isinstance(input, (StagingInput, DummyConstantInput)):
            input = StagingInput(input, towers)
    return input


def launch_train_with_config(config, trainer):
    """
    Train with a :class:`TrainConfig` and a :class:`Trainer`, to
    mimic the old training interface. It basically does the following
    3 things (and you can easily do them by yourself):

    1. Setup the :class:`InputSource` with automatic prefetching,
       for `config.data` or `config.dataflow`.
    2. Call `trainer.setup_graph` with the :class:`InputSource`,
       as well as `config.model`.
    3. Call `trainer.train` with rest of the attributes of config.

    Args:
        config (TrainConfig):
        trainer (Trainer): an instance of a SingleCostTrainer

    Examples:

    .. code-block:: python

        # with the old trainer:
        SyncMultiGPUTrainerParameterServer(config, ps_device='gpu').train()
        # with the new trainer:
        launch_train_with_config(
            config, SyncMultiGPUTrainerParameterServer(towers, ps_device='gpu'))
    """
    assert isinstance(trainer, SingleCostTrainer), trainer
    assert isinstance(config, TrainConfig), config
    assert config.model is not None
    assert config.dataflow is not None or config.data is not None

    model = config.model
    inputs_desc = model.get_inputs_desc()
    input = config.data or config.dataflow
    input = apply_default_prefetch(input, trainer, config.tower)

    if isinstance(trainer, DistributedTrainerReplicated) and \
            config.session_config is not None:
        raise ValueError(
            "Cannot set session_config for distributed training! "
            "To use a custom session config, pass it to tf.train.Server.")

    trainer.setup_graph(
        inputs_desc, input,
        model._build_graph_get_cost, model.get_optimizer)
    trainer.train(
        config.callbacks, config.monitors,
        config.session_creator, config.session_init,
        config.steps_per_epoch, config.starting_epoch, config.max_epoch)
