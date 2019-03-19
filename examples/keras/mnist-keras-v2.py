#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-keras-v2.py
# Author: Yuxin Wu

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorpack import QueueInput
from tensorpack.callbacks import ModelSaver
from tensorpack.contrib.keras import KerasModel
from tensorpack.dataflow import BatchData, MapData, dataset
from tensorpack.utils import logger

KL = keras.layers
IMAGE_SIZE = 28


def get_data():
    def f(dp):
        im = dp[0][:, :, None]
        onehot = np.eye(10)[dp[1]]
        return [im, onehot]

    train = BatchData(MapData(dataset.Mnist('train'), f), 128)
    test = BatchData(MapData(dataset.Mnist('test'), f), 256)
    return train, test


if __name__ == '__main__':
    logger.auto_set_dir('d')

    def model_func(image):
        """
        Keras model has to be created inside this function to be used with tensorpack.
        """
        M = keras.models.Sequential()
        # input_tensor have to be used here for tensorpack trainer to function properly.
        # Just use inputs[1], inputs[2] if you have multiple inputs.
        M.add(KL.InputLayer(input_tensor=image))
        M.add(KL.Conv2D(32, 3, activation='relu', padding='same'))
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
        return M

    dataset_train, dataset_test = get_data()

    M = KerasModel(
        model_func,
        input_signature=[tf.TensorSpec([None, IMAGE_SIZE, IMAGE_SIZE, 1], tf.float32, 'images')],
        target_signature=[tf.TensorSpec([None, 10], tf.float32, 'labels')],
        input=QueueInput(dataset_train))
    M.compile(
        optimizer=tf.train.AdamOptimizer(1e-3),
        loss='categorical_crossentropy',
        metrics='categorical_accuracy'
    )
    M.fit(
        validation_data=dataset_test,
        steps_per_epoch=len(dataset_train),
        callbacks=[
            ModelSaver()
        ]
    )
