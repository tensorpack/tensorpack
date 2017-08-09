#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-keras.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import sys
import argparse

import keras.layers as KL
import keras.backend as KB
from keras.models import Sequential
from keras import regularizers

"""
This is an mnist example demonstrating how to use Keras models inside tensorpack.
This way you can define models in Keras-style, and benefit from the more efficeint trainers in tensorpack.
"""

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.argtools import memoized

IMAGE_SIZE = 28


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputDesc(tf.int32, (None,), 'label'),
                ]

    @memoized   # this is necessary for sonnet/Keras to work under tensorpack
    def _build_keras_model(self):
        M = Sequential()
        M.add(KL.Conv2D(32, 3, activation='relu', input_shape=[IMAGE_SIZE, IMAGE_SIZE, 1], padding='same'))
        M.add(KL.MaxPooling2D())
        M.add(KL.Conv2D(32, 3, activation='relu', padding='same'))
        M.add(KL.Conv2D(32, 3, activation='relu', padding='same'))
        M.add(KL.MaxPooling2D())
        M.add(KL.Conv2D(32, 3, padding='same', activation='relu'))
        M.add(KL.Flatten())
        M.add(KL.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
        M.add(KL.Dropout(0.5))
        M.add(KL.Dense(10, activation=None, kernel_regularizer=regularizers.l2(1e-5)))
        return M

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.expand_dims(image, 3)

        image = image * 2 - 1   # center the pixels values at zero

        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
            M = self._build_keras_model()
            logits = M(image)
        prob = tf.nn.softmax(logits, name='prob')   # a Bx10 with probabilities

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        wrong = symbolic_functions.prediction_incorrect(logits, label, name='incorrect')
        train_error = tf.reduce_mean(wrong, name='train_error')
        summary.add_moving_summary(train_error)

        wd_cost = tf.add_n(M.losses, name='regularize_loss')    # this is how Keras manage regularizers
        self.cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, self.cost)

        # this is the keras naming
        summary.add_param_summary(('conv2d.*/kernel', ['histogram', 'rms']))

    def _get_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


# Keras needs an extra input
class KerasCallback(Callback):
    def __init__(self, isTrain):
        self._isTrain = isTrain
        self._learning_phase = KB.learning_phase()

    def _before_run(self, ctx):
        return tf.train.SessionRunArgs(
            fetches=[], feed_dict={self._learning_phase: int(self._isTrain)})


def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return train, test


def get_config():
    logger.auto_set_dir()
    dataset_train, dataset_test = get_data()

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            KerasCallback(1),   # for Keras training
            ModelSaver(),
            InferenceRunner(
                dataset_test,
                [ScalarStats('cross_entropy_loss'), ClassificationError('incorrect')],
                extra_hooks=[CallbackToHook(KerasCallback(0))]),    # for keras inference
        ],
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    if config.nr_tower > 1:
        SyncMultiGPUTrainer(config).train()
    else:
        QueueInputTrainer(config).train()
