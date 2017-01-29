#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-disturb.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import os
import sys
import argparse

from tensorpack import *
from disturb import DisturbLabel

import imp
mnist_example = imp.load_source('mnist_example',
                                os.path.join(os.path.dirname(__file__), '..', 'mnist-convnet.py'))
get_config = mnist_example.get_config


def get_data():
    dataset_train = BatchData(DisturbLabel(dataset.Mnist('train'), args.prob), 128)
    dataset_test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return dataset_train, dataset_test


mnist_example.get_data = get_data
IMAGE_SIZE = 28


class Model(mnist_example.Model):
    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.expand_dims(image, 3)

        with argscope(Conv2D, kernel_shape=5, nl=tf.nn.relu):
            logits = (LinearWrap(image)  # the starting brace is only for line-breaking
                      .Conv2D('conv0', out_channel=32, padding='VALID')
                      .MaxPooling('pool0', 2)
                      .Conv2D('conv1', out_channel=64, padding='VALID')
                      .MaxPooling('pool1', 2)
                      .FullyConnected('fc0', 512, nl=tf.nn.relu)
                      .FullyConnected('fc1', out_dim=10, nl=tf.identity)())
        prob = tf.nn.softmax(logits, name='prob')

        wrong = symbolic_functions.prediction_incorrect(logits, label)
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        wd_cost = tf.multiply(1e-5, regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')

        self.cost = tf.add_n([wd_cost, cost], name='cost')
        add_moving_summary(cost, wd_cost, self.cost)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--prob', help='disturb prob', type=float, required=True)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    QueueInputTrainer(config).train()
