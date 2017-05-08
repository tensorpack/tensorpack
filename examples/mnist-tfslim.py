#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-tfslim.py

import numpy as np
import os
import sys
import argparse

"""
MNIST ConvNet example.
about 0.6% validation error after 30 epochs.
"""

# Just import everything into current namespace
from tensorpack import *
import tensorflow as tf
import tensorflow.contrib.slim as slim

IMAGE_SIZE = 28


class Model(ModelDesc):
    def _get_inputs(self):
        """
        Define all the inputs (with type, shape, name) that
        the graph will need.
        """
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        """This function should build the model which takes the input variables
        and define self.cost at the end"""

        # inputs contains a list of input variables defined above
        image, label = inputs
        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        image = tf.expand_dims(image, 3)

        image = image * 2 - 1   # center the pixels values at zero

        is_training = get_current_tower_context().is_training
        with slim.arg_scope([slim.layers.fully_connected],
                            weights_regularizer=slim.l2_regularizer(1e-5)):
            l = slim.layers.conv2d(image, 32, [3, 3], scope='conv0')
            l = slim.layers.max_pool2d(l, [2, 2], scope='pool0')
            l = slim.layers.conv2d(l, 32, [3, 3], padding='SAME', scope='conv1')
            l = slim.layers.conv2d(l, 32, [3, 3], scope='conv2')
            l = slim.layers.max_pool2d(l, [2, 2], scope='pool1')
            l = slim.layers.conv2d(l, 32, [3, 3], scope='conv3')
            l = slim.layers.flatten(l, scope='flatten')
            l = slim.layers.fully_connected(l, 512, scope='fc0')
            l = slim.layers.dropout(l, is_training=is_training)
            logits = slim.layers.fully_connected(l, 10, activation_fn=None, scope='fc1')

        prob = tf.nn.softmax(logits, name='prob')   # a Bx10 with probabilities

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        # compute the "incorrect vector", for the callback ClassificationError to use at validation time
        wrong = symbolic_functions.prediction_incorrect(logits, label, name='incorrect')

        # This will monitor training error (in a moving_average fashion):
        # 1. write the value to tensosrboard
        # 2. write the value to stat.json
        # 3. print the value after each epoch
        train_error = tf.reduce_mean(wrong, name='train_error')
        summary.add_moving_summary(train_error)

        # slim already adds regularization to a collection, no extra handling
        self.cost = cost
        summary.add_moving_summary(cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']),
                                  ('.*/weights', ['histogram', 'rms'])  # to also work with slim
                                  )

    def _get_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return train, test


def get_config():
    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data()
    # How many iterations you want in each epoch.
    # This is the default value, don't actually need to set it in the config
    steps_per_epoch = dataset_train.size()

    # get the config which contains everything necessary in a training
    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,  # the DataFlow instance for training
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                # Calculate both the cost and the error for this DataFlow
                [ScalarStats('cross_entropy_loss'), ClassificationError('incorrect')]),
        ],
        steps_per_epoch=steps_per_epoch,
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
    SimpleTrainer(config).train()
