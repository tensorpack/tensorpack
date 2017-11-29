#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-tfslim.py

import os
import argparse
"""
MNIST ConvNet example using TensorFlow-slim.
Mostly the same as 'mnist-convnet.py',
the only differences are:
    1. use slim.layers, slim.arg_scope, etc
    2. use slim names to summarize weights
"""


from tensorpack import *
from tensorpack.dataflow import dataset
import tensorflow as tf
import tensorflow.contrib.slim as slim

IMAGE_SIZE = 28


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.expand_dims(image, 3)

        image = image * 2 - 1

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

        tf.nn.softmax(logits, name='prob')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        acc = tf.to_float(tf.nn.in_top_k(logits, label, 1))

        acc = tf.reduce_mean(acc, name='accuracy')
        summary.add_moving_summary(acc)

        self.cost = cost
        summary.add_moving_summary(cost)
        summary.add_param_summary(('.*/weights', ['histogram', 'rms']))  # slim uses different variable names

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


def get_config():
    logger.auto_set_dir()
    dataset_train, dataset_test = get_data()
    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(
                dataset_test,
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    launch_train_with_config(config, SimpleTrainer())
