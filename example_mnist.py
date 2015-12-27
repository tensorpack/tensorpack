#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: example_mnist.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


# prefer protobuf in user-namespace
import sys
import os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python2.7/site-packages'))

import tensorflow as tf
import numpy as np

from layers import *
from utils import *
from dataflow.dataset import Mnist
from dataflow import *

IMAGE_SIZE = 28
LOG_DIR = 'train_log'

def get_model(inputs):
    """
    Args:
        inputs: a list of input variable,
        e.g.: [image_var, label_var] with:
            image_var: bx28x28
            label_var: bx1 integer
    Returns:
        (outputs, cost)
        outputs: a list of output variable
        cost: scalar variable
    """
    # use this variable in dropout! Tensorpack will automatically set it to 1 at test time
    keep_prob = tf.placeholder(tf.float32, shape=tuple(), name=DROPOUT_PROB_OP_NAME)

    image, label = inputs

    image = tf.reshape(image, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    conv0 = Conv2D('conv0', image, out_channel=32, kernel_shape=5,
                  padding='valid')
    conv0 = tf.nn.relu(conv0)
    pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
    conv1 = Conv2D('conv1', pool0, out_channel=40, kernel_shape=3, padding='valid')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    feature = batch_flatten(pool1)

    fc0 = FullyConnected('fc0', feature, 1024)
    fc0 = tf.nn.relu(fc0)
    fc0 = tf.nn.dropout(fc0, keep_prob)

    fc1 = FullyConnected('lr', fc0, out_dim=10)
    prob = tf.nn.softmax(fc1, name='output')

    y = one_hot(label, 10)
    cost = tf.nn.softmax_cross_entropy_with_logits(fc1, y)
    cost = tf.reduce_mean(cost, name='cross_entropy_loss')
    tf.add_to_collection(COST_VARS_KEY, cost)

    # compute the number of correctly classified samples, for ValidationAccuracy to use at test time
    correct = tf.equal(
        tf.cast(tf.argmax(prob, 1), tf.int32), label)
    correct = tf.cast(correct, tf.float32)
    nr_correct = tf.reduce_sum(correct, name='correct')

    # monitor training accuracy
    tf.add_to_collection(
        SUMMARY_VARS_KEY,
        tf.reduce_mean(correct, name='train_accuracy'))

    # weight decay on all W of fc layers
    wd_cost = tf.mul(1e-4,
                     regularize_cost('fc.*/W', tf.nn.l2_loss),
                     name='regularize_loss')
    tf.add_to_collection(COST_VARS_KEY, wd_cost)

    return [prob, nr_correct], tf.add_n(tf.get_collection(COST_VARS_KEY), name='cost')

def main():
    BATCH_SIZE = 128
    with tf.Graph().as_default():
        dataset_train = BatchData(Mnist('train'), BATCH_SIZE)
        dataset_test = BatchData(Mnist('test'), 256, remainder=True)

        sess_config = tf.ConfigProto()
        sess_config.device_count['GPU'] = 1

        # prepare model
        image_var = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE), name='input')
        label_var = tf.placeholder(tf.int32, shape=(None,), name='label')
        input_vars = [image_var, label_var]
        output_vars, cost_var = get_model(input_vars)

        config = dict(
            dataset_train=dataset_train,
            optimizer=tf.train.AdamOptimizer(1e-4),
            callbacks=[
                ValidationAccuracy(
                    dataset_test,
                    prefix='test'),
                PeriodicSaver(LOG_DIR, period=1),
                SummaryWriter(LOG_DIR, histogram_regex='.*/W'),
            ],
            session_config=sess_config,
            inputs=input_vars,
            outputs=output_vars,
            cost=cost_var,
            max_epoch=100,
        )
        from train import start_train
        start_train(config)


if __name__ == '__main__':
    main()
