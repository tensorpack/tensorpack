#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-keras.py
# Author: Yuxin Wu

import tensorflow as tf
from tensorflow import keras
from contextlib import contextmanager

from tensorpack import *
from tensorpack.contrib.keras import KerasPhaseCallback
from tensorpack.dataflow import dataset
from tensorpack.utils.argtools import memoized
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.tower import get_current_tower_context

KL = keras.layers

"""
This is an mnist example demonstrating how to use Keras symbolic function inside tensorpack.
This way you can define models in Keras-style, and benefit from the more efficeint trainers in tensorpack.

Note: this example does not work for replicated-style data-parallel trainers, so may be less efficient
for some models.
"""

IMAGE_SIZE = 28


# Work around a Keras issue: it append name scopes to variable names..
# May not work well if you use Keras layers inside other name scopes.
@contextmanager
def clear_tower0_name_scope():
    ns = tf.get_default_graph().get_name_scope()
    if ns == 'tower0':
        with tf.name_scope('/'):
            yield
    else:
        yield


@memoized  # this is necessary for sonnet/keras to work under tensorpack
def get_keras_model():
    with clear_tower0_name_scope():
        M = keras.models.Sequential()
        M.add(KL.Conv2D(32, 3, activation='relu', padding='same'))
        M.add(KL.MaxPooling2D())
        M.add(KL.Conv2D(32, 3, activation='relu', padding='same'))
        M.add(KL.Conv2D(32, 3, activation='relu', padding='same'))
        M.add(KL.MaxPooling2D())
        M.add(KL.Conv2D(32, 3, padding='same', activation='relu'))
        M.add(KL.Flatten())
        M.add(KL.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5)))
        M.add(KL.Dropout(rate=0.5))
        M.add(KL.Dense(10, activation=None, kernel_regularizer=keras.regularizers.l2(1e-5)))
    return M


class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    def build_graph(self, image, label):
        image = tf.expand_dims(image, 3) * 2 - 1
        ctx = get_current_tower_context()

        M = get_keras_model()
        logits = M(image)
        if ctx.is_main_training_tower:
            for op in M.updates:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)

        # build cost function by tensorflow
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        # for tensorpack validation
        acc = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32)
        acc = tf.reduce_mean(acc, name='accuracy')
        summary.add_moving_summary(acc)

        wd_cost = tf.add_n(M.losses, name='regularize_loss')    # this is how Keras manage regularizers
        cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost)
        return cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return train, test


if __name__ == '__main__':
    logger.auto_set_dir()
    dataset_train, dataset_test = get_data()

    cfg = TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            KerasPhaseCallback(True),   # for Keras training
            ModelSaver(),
            InferenceRunner(
                dataset_test,
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        max_epoch=100,
    )

    if get_num_gpu() <= 1:
        # single GPU:
        launch_train_with_config(cfg, SimpleTrainer())
    else:
        # multi GPU:
        launch_train_with_config(cfg, SyncMultiGPUTrainerParameterServer(2))
        # "Replicated" multi-gpu trainer is not supported for Keras model
        # since Keras does not respect variable scopes.
