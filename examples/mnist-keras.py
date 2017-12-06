#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-keras.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from tensorflow import keras
KL = keras.layers

"""
This is an mnist example demonstrating how to use Keras symbolic function inside tensorpack.
This way you can define models in Keras-style, and benefit from the more efficeint trainers in tensorpack.
"""


from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.argtools import memoized
from tensorpack.contrib.keras import KerasPhaseCallback

IMAGE_SIZE = 28


@memoized        # this is necessary for sonnet/Keras to work under tensorpack
def get_keras_model():
    M = keras.models.Sequential()
    M.add(KL.Conv2D(32, 3, activation='relu', input_shape=[IMAGE_SIZE, IMAGE_SIZE, 1], padding='same'))
    M.add(KL.MaxPooling2D())
    M.add(KL.Conv2D(32, 3, activation='relu', padding='same'))
    M.add(KL.Conv2D(32, 3, activation='relu', padding='same'))
    M.add(KL.MaxPooling2D())
    M.add(KL.Conv2D(32, 3, padding='same', activation='relu'))
    M.add(KL.Flatten())
    M.add(KL.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5)))
    M.add(KL.Dropout(0.5))
    M.add(KL.Dense(10, activation=None, kernel_regularizer=keras.regularizers.l2(1e-5)))
    return M


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.expand_dims(image, 3) * 2 - 1

        M = get_keras_model()
        logits = M(image)
        # build cost function by tensorflow
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        # for tensorpack validation
        acc = tf.to_float(tf.nn.in_top_k(logits, label, 1))
        acc = tf.reduce_mean(acc, name='accuracy')
        summary.add_moving_summary(acc)

        wd_cost = tf.add_n(M.losses, name='regularize_loss')    # this is how Keras manage regularizers
        self.cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(self.cost, wd_cost)

    def _get_optimizer(self):
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

    launch_train_with_config(cfg, QueueInputTrainer())
