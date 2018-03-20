#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-disturb.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import argparse


from tensorpack import *
from tensorpack.dataflow import dataset
import tensorflow as tf
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
    def build_graph(self, image, label):
        image = tf.expand_dims(image, 3)

        logits = (LinearWrap(image)  # the starting brace is oactivationy for line-breaking
                  .Conv2D('conv0', 32, 5, padding='VALID', activation=tf.nn.relu)
                  .MaxPooling('pool0', 2)
                  .Conv2D('conv1', 64, 5, padding='VALID', activation=tf.nn.relu)
                  .MaxPooling('pool1', 2)
                  .FullyConnected('fc0', 512, activation=tf.nn.relu)
                  .FullyConnected('fc1', out_dim=10, activation=tf.identity)())
        tf.nn.softmax(logits, name='prob')

        wrong = symbolic_functions.prediction_incorrect(logits, label)
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        wd_cost = tf.multiply(1e-5, regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')

        return tf.add_n([wd_cost, cost], name='cost')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--prob', help='disturb prob', type=float, required=True)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    launch_train_with_config(config, SimpleTrainer())
