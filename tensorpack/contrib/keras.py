#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: keras.py

import tensorflow as tf
import six
from tensorflow import keras
from tensorflow.python.keras import metrics as metrics_module


from ..models.regularize import regularize_cost_from_collection
from ..graph_builder import InputDesc
from ..tfutils.tower import get_current_tower_context
# from ..tfutils.collection import freeze_collection    # TODO freeze UPDATE_OPS in replicated
from ..callbacks import (
    Callback, InferenceRunner, CallbackToHook,
    ScalarStats)
from ..tfutils.summary import add_moving_summary
from ..utils.gpu import get_nr_gpu
from ..train import Trainer, SimpleTrainer, SyncMultiGPUTrainerParameterServer


__all__ = ['KerasPhaseCallback', 'setup_keras_trainer', 'KerasModel']


class KerasModelCaller(object):
    """
    Keras model doesn't support vs reuse.
    This is hack to mimic reuse.
    """
    def __init__(self, get_model):
        self.get_model = get_model

        self.cached_model = None

    def __call__(self, input_tensors):
        reuse = tf.get_variable_scope().reuse
        if self.cached_model is None:
            assert not reuse
            self.cached_model = self.get_model(input_tensors)
            return self.cached_model.outputs

        if reuse:
            return self.cached_model.call(input_tensors)
        else:
            M = self.get_model(input_tensors)
            return M.outputs


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
        trainer, get_model, input,
        optimizer, loss, metrics=None):
    """
    Args:
        trainer (SingleCostTrainer):
        get_model ( -> keras.model.Model):
        input (InputSource):
        optimizer (tf.tarin.Optimizer):
        loss, metrics: list of strings
    """
    assert isinstance(optimizer, tf.train.Optimizer), optimizer

    G_tmp = tf.Graph()  # we need the model instance to know metadata about inputs/outputs
    with G_tmp.as_default():
        M_tmp = get_model([None])   # TODO use a proxy with Nones
        inputs_desc = [InputDesc(t.dtype, t.shape.as_list(), 'input{}'.format(i))
                       for i, t in enumerate(M_tmp.inputs)]
        outputs_desc = [InputDesc(t.dtype, t.shape.as_list(), 'output{}'.format(i))
                        for i, t in enumerate(M_tmp.outputs)]
        nr_inputs = len(inputs_desc)
    del G_tmp, M_tmp

    model_caller = KerasModelCaller(get_model)

    def get_cost(*inputs):
        assert len(inputs) == len(inputs_desc) + len(outputs_desc), \
            "Input source size {} != {} + {}".format(len(inputs), len(inputs_desc), len(outputs_desc))
        ctx = get_current_tower_context()
        input_tensors = list(inputs[:nr_inputs])
        target_tensors = list(inputs[nr_inputs:])
        # TODO mapping between target tensors & output tensors

        outputs = model_caller(input_tensors)

        if isinstance(outputs, tf.Tensor):
            outputs = [outputs]
        assert len(outputs) == len(target_tensors), \
            "len({}) != len({})".format(str(outputs), str(target_tensors))
        assert len(outputs) == len(loss), \
            "len({}) != len({})".format(str(outputs), str(loss))

        # TODO more losses
        with tf.name_scope('keras_loss'):
            loss_fn = keras.losses.get(loss[0])
            loss_opt = loss_fn(target_tensors[0], outputs[0])
        loss_opt = tf.reduce_mean(loss_opt, name=loss[0])

        loss_reg = regularize_cost_from_collection()
        if loss_reg is not None:
            total_loss = tf.add(loss_opt, loss_reg, name='total_loss')
            add_moving_summary(loss_opt, loss_reg, total_loss)
        else:
            add_moving_summary(loss_opt)
            total_loss = tf.identity(loss_opt, name='total_loss')

        if metrics and (ctx.is_main_training_tower or not ctx.is_training):
            # for list: one metric for each output
            metric_tensors = []
            for oid, metric_name in enumerate(metrics):
                output_tensor = outputs[oid]
                target_tensor = target_tensors[oid]  # TODO may not have the same mapping?
                with tf.name_scope('keras_metric'):  # TODO ns reuse
                    metric_fn = metrics_module.get(metric_name)
                    metric_tensor = metric_fn(target_tensor, output_tensor)
                metric_tensor = tf.reduce_mean(metric_tensor, name=metric_name)
                # check name conflict here
                metric_tensors.append(metric_tensor)
            add_moving_summary(*metric_tensors)

        return total_loss

    trainer.setup_graph(
        inputs_desc + outputs_desc,
        input,
        get_cost,
        lambda: optimizer)
    if model_caller.cached_model.uses_learning_phase:
        trainer.register_callback(KerasPhaseCallback(True))


class KerasModel(object):
    def __init__(self, get_model, input, trainer=None):
        """
        Args:
            get_model ( -> keras.model.Model):
            input (InputSource):
            trainer (Trainer): the default will check the number of available
                GPUs and use them all.
        """
        self.get_model = get_model
        if trainer is None:
            nr_gpu = get_nr_gpu()
            if nr_gpu <= 1:
                trainer = SimpleTrainer()
            else:
                trainer = SyncMultiGPUTrainerParameterServer(nr_gpu)
        assert isinstance(trainer, Trainer), trainer

        self.input = input
        self.trainer = trainer

    def compile(self, optimizer, loss, metrics=None):
        """
        Args:
            optimizer (tf.train.Optimizer):
            loss, metrics: string or list of strings
        """
        if isinstance(loss, six.string_types):
            loss = [loss]
        if metrics is None:
            metrics = []
        if isinstance(metrics, six.string_types):
            metrics = [metrics]

        self._stats_to_inference = loss + metrics
        setup_keras_trainer(
            self.trainer, get_model=self.get_model,
            input=self.input,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)

    def fit(self, validation_data=None, **kwargs):
        """
        Args:
            validation_data (DataFlow or InputSource): to be used for inference.
            kwargs: same as `self.trainer.train_with_defaults`.
        """
        callbacks = kwargs.pop('callbacks', [])
        if validation_data is not None:
            callbacks.append(
                InferenceRunner(
                    validation_data, ScalarStats(self._stats_to_inference + ['total_loss'])))
        self.trainer.train_with_defaults(callbacks=callbacks, **kwargs)
