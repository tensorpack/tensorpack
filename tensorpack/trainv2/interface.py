#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: interface.py

import tensorflow as tf

from ..input_source import (
    FeedInput, QueueInput, StagingInputWrapper, DummyConstantInput)

from ..train.config import TrainConfig
from .base import SingleCostTrainer
from .trainers import SimpleTrainer, DistributedTrainerReplicated

__all__ = ['launch_train_with_config', 'TrainConfig']


def _maybe_gpu_prefetch(input, towers, gpu_prefetch):
    # seem to only improve on >1 GPUs
    if len(towers) > 1 and gpu_prefetch:
        assert tf.test.is_gpu_available()

        if not isinstance(input, (StagingInputWrapper, DummyConstantInput)):
            input = StagingInputWrapper(input, towers)
    return input


def launch_train_with_config(config, trainer):
    """
    To mimic the old training interface, with a trainer and a config.

    Args:
        config (TrainConfig):
        trainer (Trainer): an instance of the new trainer

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
    input = config.data

    # some check & input wrappers to mimic same behavior of the old trainer interface
    if input is None:
        if type(trainer) == SimpleTrainer:
            input = FeedInput(config.dataflow)
        else:
            input = QueueInput(config.dataflow)

    if config.nr_tower > 1:
        assert not isinstance(trainer, SimpleTrainer)
        input = _maybe_gpu_prefetch(input, config.tower, True)

    if isinstance(trainer, DistributedTrainerReplicated) and \
            config.session_config is not None:
        raise ValueError(
            "Cannot set session_config for distributed training! "
            "To use a custom session config, pass it to tf.train.Server.")

    trainer.setup_graph(
        inputs_desc, input,
        model.build_graph_get_cost, model.get_optimizer)
    trainer.train(
        config.callbacks,
        config.monitors,
        config.session_creator,
        config.session_init,
        config.steps_per_epoch,
        config.starting_epoch,
        config.max_epoch)
