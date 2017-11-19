#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-keras-v2.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf

from tensorflow import keras
KL = keras.layers


from tensorpack.input_source import QueueInput
from tensorpack.dataflow import dataset, BatchData, MapData
from tensorpack.utils import logger
from tensorpack.contrib.keras import KerasModel

IMAGE_SIZE = 28


def get_data():
    def f(dp):
        im = dp[0][:, :, None]
        onehot = np.zeros(10, dtype='int32')
        onehot[dp[1]] = 1
        return [im, onehot]

    train = BatchData(MapData(dataset.Mnist('train'), f), 128)
    test = BatchData(MapData(dataset.Mnist('test'), f), 256)
    return train, test


if __name__ == '__main__':
    logger.auto_set_dir()
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
    M.add(KL.Activation('softmax'))

    dataset_train, dataset_test = get_data()

    M = KerasModel(M, QueueInput(dataset_train))
    M.compile(
        optimizer=tf.train.AdamOptimizer(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    M.fit(
        validation_data=dataset_test,
        steps_per_epoch=dataset_train.size(),
    )
