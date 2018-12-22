#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-tfslim.py

"""
MNIST ConvNet example using TensorFlow-slim.
Mostly the same as 'mnist-convnet.py',
the only differences are:
    1. use slim.layers, slim.arg_scope, etc
    2. use slim names to summarize weights
"""


import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorpack import *
from tensorpack.dataflow import dataset

IMAGE_SIZE = 28


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, image, label):
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

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        acc = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32)

        acc = tf.reduce_mean(acc, name='accuracy')
        summary.add_moving_summary(acc)

        summary.add_moving_summary(cost)
        summary.add_param_summary(('.*/weights', ['histogram', 'rms']))  # slim uses different variable names
        return cost + regularize_cost_from_collection()

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

    config = TrainConfig(
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
    launch_train_with_config(config, SimpleTrainer())
