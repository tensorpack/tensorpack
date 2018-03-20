#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-tflayers.py

import os
import argparse
import tensorflow as tf
"""
MNIST ConvNet example using tf.layers
Mostly the same as 'mnist-convnet.py',
the only differences are:
    1. use tf.layers
    2. use tf.layers variable names to summarize weights
"""


# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary, get_current_tower_context
from tensorpack.dataflow import dataset

IMAGE_SIZE = 28


class Model(ModelDesc):
    def inputs(self):
        """
        Define all the inputs (with type, shape, name) that
        the graph will need.
        """
        return [tf.placeholder(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, image, label):
        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        image = tf.expand_dims(image, 3)

        image = image * 2 - 1   # center the pixels values at zero

        l = tf.layers.conv2d(image, 32, 3, padding='same', activation=tf.nn.relu, name='conv0')
        l = tf.layers.max_pooling2d(l, 2, 2, padding='valid')
        l = tf.layers.conv2d(l, 32, 3, padding='same', activation=tf.nn.relu, name='conv1')
        l = tf.layers.conv2d(l, 32, 3, padding='same', activation=tf.nn.relu, name='conv2')
        l = tf.layers.max_pooling2d(l, 2, 2, padding='valid')
        l = tf.layers.conv2d(l, 32, 3, padding='same', activation=tf.nn.relu, name='conv3')
        l = tf.layers.flatten(l)
        l = tf.layers.dense(l, 512, activation=tf.nn.relu, name='fc0')
        l = tf.layers.dropout(l, rate=0.5,
                              training=get_current_tower_context().is_training)
        logits = tf.layers.dense(l, 10, activation=tf.identity, name='fc1')

        tf.nn.softmax(logits, name='prob')   # a Bx10 with probabilities

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error (in a moving_average fashion):
        # 1. write the value to tensosrboard
        # 2. write the value to stat.json
        # 3. print the value after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('fc.*/kernel', tf.nn.l2_loss),
                              name='regularize_loss')
        total_cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, total_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/kernel', ['histogram', 'rms']))
        return total_cost

    def optimizer(self):
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
            MaxSaver('validation_accuracy'),  # save the model with highest accuracy (prefix 'validation_')
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
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

    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    # SimpleTrainer is slow, this is just a demo.
    # You can use QueueInputTrainer instead
    launch_train_with_config(config, SimpleTrainer())
