# -*- coding: utf-8 -*-
# File: interface.py

from ..compat import tfv1
from ..input_source import DummyConstantInput, FeedfreeInput, FeedInput, InputSource, QueueInput, StagingInput
from ..utils import logger
from ..compat import is_tfv2
from .config import TrainConfig
from .tower import SingleCostTrainer
from .trainers import SimpleTrainer

__all__ = ['launch_train_with_config']


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
    present the simple and old training interface. It basically does the following
    3 things (and you can easily do them by yourself if you need more control):

    1. Setup the input with automatic prefetching heuristics,
       from `config.data` or `config.dataflow`.
    2. Call `trainer.setup_graph` with the input as well as `config.model`.
    3. Call `trainer.train` with rest of the attributes of config.

    See the `related tutorial
    <https://tensorpack.readthedocs.io/tutorial/training-interface.html#with-modeldesc-and-trainconfig>`_
    to learn more.

    Args:
        config (TrainConfig):
        trainer (Trainer): an instance of :class:`SingleCostTrainer`.

    Example:

    .. code-block:: python

        launch_train_with_config(
            config, SyncMultiGPUTrainerParameterServer(8, ps_device='gpu'))
    """
    if is_tfv2():
        tfv1.disable_eager_execution()

    assert isinstance(trainer, SingleCostTrainer), trainer
    assert isinstance(config, TrainConfig), config
    assert config.model is not None
    assert config.dataflow is not None or config.data is not None

    model = config.model
    input = config.data or config.dataflow
    input = apply_default_prefetch(input, trainer)

    # This is the only place where the `ModelDesc` abstraction is useful.
    # We should gradually stay away from this unuseful abstraction.
    # TowerFunc is a better abstraction (similar to tf.function in the future)
    trainer.setup_graph(
        model.get_input_signature(), input,
        model.build_graph, model.get_optimizer)
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
    coll = tfv1.get_collection(tfv1.GraphKeys.REGULARIZATION_LOSSES)
    unconsumed_reg = []
    for c in coll:
        if len(c.consumers()) == 0:
            unconsumed_reg.append(c)
    if unconsumed_reg:
        logger.warn("The following tensors appear in REGULARIZATION_LOSSES collection but have no "
                    "consumers! You may have forgotten to add regularization to total cost.")
        logger.warn("Unconsumed regularization: {}".format(', '.join([x.name for x in unconsumed_reg])))
