#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: keras.py

import tensorflow as tf
from six.moves import zip
import keras

from ..graph_builder import InputDesc
from ..tfutils.tower import get_current_tower_context
from ..tfutils.collection import freeze_collection
from ..callbacks import Callback, InferenceRunner, CallbackToHook
from ..tfutils.summary import add_moving_summary


# Keras needs an extra input if learning_phase is used by the model
# This cb will be used by
# 1. trainer with isTrain=True
# 2. InferenceRunner with isTrain=False, in the form of hooks
class KerasPhaseCallback(Callback):
    def __init__(self, isTrain):
        assert isinstance(isTrain, bool), isTrain
        self._isTrain = isTrain
        self._learning_phase = keras.backend.learning_phase()

    def _setup_graph(self):
        # HACK
        cbs = self.trainer._callbacks.cbs
        for cb in cbs:
            if isinstance(cb, InferenceRunner):
                h = CallbackToHook(KerasPhaseCallback(False))
                cb.register_hook(h)

    def _before_run(self, ctx):
        return tf.train.SessionRunArgs(
            fetches=[], feed_dict={self._learning_phase: int(self._isTrain)})


def setup_keras_trainer(
        trainer, model, input,
        optimizer, loss, metrics=None):
    """
    Args:
        trainer (SingleCostTrainer):
        model (keras.model.Model):
        input (InputSource):
        optimizer (tf.tarin.Optimizer):
        loss, metrics: same as in `keras.model.Model.compile()`.
    """
    assert isinstance(optimizer, tf.train.Optimizer), optimizer
    inputs_desc = [InputDesc.from_tensor(t) for t in model.inputs]
    outputs_desc = [InputDesc.from_tensor(t) for t in model.outputs]
    nr_inputs = len(inputs_desc)

    # clear the collection
    del tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)[:]

    def get_cost(*inputs):
        ctx = get_current_tower_context()
        assert ctx.is_main_training_tower or not ctx.has_own_variables
        input_tensors = list(inputs[:nr_inputs])
        target_tensors = list(inputs[nr_inputs:])

        # Keras check and do weird things if target is a placeholder..
        # Use tf.identity so it's not a placeholder.
        target_tensors = [tf.identity(t) for t in target_tensors]

        input_keras_tensors = [keras.layers.Input(tensor=t) for t in input_tensors]
        outputs = model(input_keras_tensors)

        M = keras.models.Model(input_tensors, outputs)

        with freeze_collection([tf.GraphKeys.TRAINABLE_VARIABLES]):
            # Keras optimizer mistakenly creates TRAINABLE_VARIABLES ...
            M.compile(
                optimizer=optimizer, loss=loss,
                target_tensors=target_tensors,
                metrics=metrics)

        add_moving_summary(tf.identity(M.total_loss, name='total_loss'))

        assert len(M.metrics) == len(M.metrics_tensors)
        for name, tensor in zip(M.metrics, M.metrics_tensors):
            add_moving_summary(tf.identity(tensor, name=name))
        # tensorpack requires TRAINABLE_VARIABLES created inside tower
        if ctx.is_main_training_tower:
            for p in M.weights:
                tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, p)
        return M.total_loss

    trainer.setup_graph(
        inputs_desc + outputs_desc,
        input,
        get_cost,
        lambda: optimizer)
    if model.uses_learning_phase:
        trainer.register_callback(KerasPhaseCallback(True))
