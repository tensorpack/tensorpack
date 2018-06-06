# -*- coding: utf-8 -*-
# File: keras.py

import tensorflow as tf
import six
from tensorflow import keras
from tensorflow.python.keras import metrics as metrics_module

from ..models.regularize import regularize_cost_from_collection
from ..train import Trainer, SimpleTrainer, SyncMultiGPUTrainerParameterServer
from ..train.trainers import DistributedTrainerBase
from ..train.interface import apply_default_prefetch
from ..callbacks import (
    Callback, InferenceRunnerBase, InferenceRunner, CallbackToHook,
    ScalarStats)

from ..tfutils.common import get_op_tensor_name
from ..tfutils.collection import backup_collection, restore_collection
from ..tfutils.tower import get_current_tower_context
from ..tfutils.scope_utils import cached_name_scope
from ..tfutils.summary import add_moving_summary
from ..utils.gpu import get_nr_gpu
from ..utils import logger


__all__ = ['KerasPhaseCallback', 'setup_keras_trainer', 'KerasModel']


TOTAL_LOSS_NAME = 'total_loss'


def _check_name(tensor, name):
    tensorname = get_op_tensor_name(tensor.name)[0]
    assert tensorname.split('/')[-1] == name, \
        "{} does not match {}, you may have name conflict somewhere!".format(tensor.name, name)


class KerasModelCaller(object):
    """
    Keras model doesn't support variable scope reuse.
    This is a hack to mimic reuse.
    """
    def __init__(self, get_model):
        self.get_model = get_model

        self.cached_model = None

    def __call__(self, input_tensors):
        """
        Args:
            input_tensors ([tf.Tensor])
        Returns:
            output tensors of this tower, evaluated with the input tensors.
        """
        reuse = tf.get_variable_scope().reuse

        old_trainable_names = set([x.name for x in tf.trainable_variables()])
        trainable_backup = backup_collection([tf.GraphKeys.TRAINABLE_VARIABLES])
        update_ops_backup = backup_collection([tf.GraphKeys.UPDATE_OPS])

        def post_process_model(model):
            added_trainable_names = set([x.name for x in tf.trainable_variables()])
            restore_collection(trainable_backup)

            for v in model.weights:
                # In Keras, the collection is not respected and could contain non-trainable vars.
                # We put M.weights into the collection instead.
                if v.name not in old_trainable_names and v.name in added_trainable_names:
                    tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, v)
            new_trainable_names = set([x.name for x in tf.trainable_variables()])

            for n in added_trainable_names:
                if n not in new_trainable_names:
                    logger.warn("Keras created trainable variable '{}' which is actually not trainable. "
                                "This was automatically corrected by tensorpack.".format(n))

            # Keras models might not use this collection at all (in some versions).
            # This is a BC-breaking change of tf.keras: https://github.com/tensorflow/tensorflow/issues/19643
            restore_collection(update_ops_backup)
            for op in model.updates:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)

        if self.cached_model is None:
            assert not reuse
            model = self.cached_model = self.get_model(*input_tensors)
            outputs = model.outputs
        elif reuse:
            # use the cached Keras model to mimic reuse
            # NOTE: ctx.is_training won't be useful inside model,
            # because inference will always use the cached Keras model
            model = self.cached_model
            outputs = model.call(input_tensors)
        else:
            # create new Keras model if not reuse
            model = self.get_model(*input_tensors)
            outputs = model.outputs

        post_process_model(model)

        return outputs


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
        logger.info("Using Keras learning phase {} in the graph!".format(
            self._learning_phase.name))
        cbs = self.trainer._callbacks.cbs
        for cb in cbs:
            # XXX HACK
            if isinstance(cb, InferenceRunnerBase):
                h = CallbackToHook(KerasPhaseCallback(False))
                cb.register_hook(h)

    def _before_run(self, ctx):
        return tf.train.SessionRunArgs(
            fetches=[], feed_dict={self._learning_phase: int(self._isTrain)})


def setup_keras_trainer(
        trainer, get_model,
        inputs_desc, targets_desc,
        input, optimizer, loss, metrics):
    """
    Args:
        trainer (SingleCostTrainer):
        get_model (input1, input2, ... -> keras.model.Model):
            Takes tensors and returns a Keras model. Will be part of the tower function.
        input (InputSource):
        optimizer (tf.train.Optimizer):
        loss, metrics: list of strings
    """
    assert isinstance(optimizer, tf.train.Optimizer), optimizer
    assert isinstance(loss, list), loss
    assert len(loss) >= 1, "No loss was given!"
    assert isinstance(metrics, list), metrics
    model_caller = KerasModelCaller(get_model)

    nr_inputs = len(inputs_desc)

    def get_cost(*inputs):
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

        loss_tensors = []
        for idx, loss_name in enumerate(loss):
            with cached_name_scope('keras_loss', top_level=False):
                loss_fn = keras.losses.get(loss_name)
                curr_loss = loss_fn(target_tensors[idx], outputs[idx])
            curr_loss = tf.reduce_mean(curr_loss, name=loss_name)
            _check_name(curr_loss, loss_name)
            loss_tensors.append(curr_loss)

        loss_reg = regularize_cost_from_collection()
        if loss_reg is not None:
            total_loss = tf.add_n(loss_tensors + [loss_reg], name=TOTAL_LOSS_NAME)
            add_moving_summary(loss_reg, total_loss, *loss_tensors)
        else:
            total_loss = tf.add_n(loss_tensors, name=TOTAL_LOSS_NAME)
            add_moving_summary(total_loss, *loss_tensors)

        if metrics and (ctx.is_main_training_tower or not ctx.is_training):
            # for list: one metric for each output
            metric_tensors = []
            for oid, metric_name in enumerate(metrics):
                output_tensor = outputs[oid]
                target_tensor = target_tensors[oid]  # TODO may not have the same mapping?
                with cached_name_scope('keras_metric', top_level=False):
                    metric_fn = metrics_module.get(metric_name)
                    metric_tensor = metric_fn(target_tensor, output_tensor)
                metric_tensor = tf.reduce_mean(metric_tensor, name=metric_name)
                _check_name(metric_tensor, metric_name)
                # check name conflict here
                metric_tensors.append(metric_tensor)
            add_moving_summary(*metric_tensors)

        return total_loss

    trainer.setup_graph(
        inputs_desc + targets_desc,
        input,
        get_cost,
        lambda: optimizer)
    if model_caller.cached_model.uses_learning_phase:
        trainer.register_callback(KerasPhaseCallback(True))


class KerasModel(object):
    def __init__(self, get_model, inputs_desc, targets_desc,
                 input, trainer=None):
        """
        Args:
            get_model (input1, input2, ... -> keras.model.Model):
                Takes tensors and returns a Keras model. Will be part of the tower function.
            inputs_desc ([InputDesc]):
            targets_desc ([InputDesc]):
            input (InputSource | DataFlow):
            trainer (Trainer): the default will check the number of available
                GPUs and use them all.
        """
        self.get_model = get_model
        self.inputs_desc = inputs_desc
        self.targets_desc = targets_desc
        if trainer is None:
            nr_gpu = get_nr_gpu()
            if nr_gpu <= 1:
                trainer = SimpleTrainer()
            else:
                # the default multi-gpu trainer
                trainer = SyncMultiGPUTrainerParameterServer(nr_gpu)
        assert isinstance(trainer, Trainer), trainer
        assert not isinstance(trainer, DistributedTrainerBase)

        self.input = apply_default_prefetch(input, trainer)
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

        self._stats_to_inference = loss + metrics + [TOTAL_LOSS_NAME]
        setup_keras_trainer(
            self.trainer, get_model=self.get_model,
            inputs_desc=self.inputs_desc, targets_desc=self.targets_desc,
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
            callbacks.append(InferenceRunner(
                validation_data, ScalarStats(self._stats_to_inference)))
        self.trainer.train_with_defaults(callbacks=callbacks, **kwargs)
