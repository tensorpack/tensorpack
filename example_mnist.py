#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: example_mnist.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

# use user-space protobuf
import sys
import os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python2.7/site-packages'))

import tensorflow as tf
import numpy as np
import os

from utils import logger
from layers import *
from utils import *
from dataflow.dataset import Mnist
from dataflow import *

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
    keep_prob = tf.get_default_graph().get_tensor_by_name(DROPOUT_PROB_VAR_NAME)

    image, label = inputs

    image = tf.expand_dims(image, 3)
    conv0 = Conv2D('conv0', image, out_channel=32, kernel_shape=5,
                  padding='valid')
    pool0 = MaxPooling('pool0', conv0, 2)
    conv1 = Conv2D('conv1', pool0, out_channel=40, kernel_shape=3)
    pool1 = MaxPooling('pool1', conv1, 2)

    fc0 = FullyConnected('fc0', pool1, 1024)
    fc0 = tf.nn.dropout(fc0, keep_prob)

    # fc will have activation summary by default. disable this for the output layer
    fc1 = FullyConnected('fc1', fc0, out_dim=10,
                         summary_activation=False, nl=tf.identity)
    prob = tf.nn.softmax(fc1, name='output')

    y = one_hot(label, 10)
    cost = tf.nn.softmax_cross_entropy_with_logits(fc1, y)
    cost = tf.reduce_mean(cost, name='cross_entropy_loss')
    tf.add_to_collection(COST_VARS_KEY, cost)

    # compute the number of failed samples, for ValidationErro to use at test time
    wrong = tf.not_equal(
        tf.cast(tf.argmax(prob, 1), tf.int32), label)
    wrong = tf.cast(wrong, tf.float32)
    nr_wrong = tf.reduce_sum(wrong, name='wrong')

    # monitor training accuracy
    tf.add_to_collection(
        SUMMARY_VARS_KEY,
        tf.sub(1.0, tf.reduce_mean(wrong), name='train_error'))

    # weight decay on all W of fc layers
    wd_cost = tf.mul(1e-4,
                     regularize_cost('fc.*/W', tf.nn.l2_loss),
                     name='regularize_loss')
    tf.add_to_collection(COST_VARS_KEY, wd_cost)

    return [prob, nr_wrong], tf.add_n(tf.get_collection(COST_VARS_KEY), name='cost')

def get_config():
    IMAGE_SIZE = 28
    LOG_DIR = 'train_log'
    BATCH_SIZE = 128
    logger.set_file(os.path.join(LOG_DIR, 'training.log'))

    dataset_train = BatchData(Mnist('train'), BATCH_SIZE)
    dataset_test = BatchData(Mnist('test'), 256, remainder=True)

    sess_config = tf.ConfigProto()
    sess_config.device_count['GPU'] = 1

    # prepare model
    image_var = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE), name='input')
    label_var = tf.placeholder(tf.int32, shape=(None,), name='label')
    input_vars = [image_var, label_var]
    output_vars, cost_var = get_model(input_vars)
    add_histogram_summary('.*/W') # monitor histogram of all W

    global_step_var = tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    lr = tf.train.exponential_decay(
        learning_rate=1e-4,
        global_step=global_step_var,
        decay_steps=dataset_train.size() * 50,
        decay_rate=0.1, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return dict(
        dataset_train=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=[
            SummaryWriter(LOG_DIR),
            ValidationError(dataset_test, prefix='test'),
            PeriodicSaver(LOG_DIR),
        ],
        session_config=sess_config,
        inputs=input_vars,
        outputs=output_vars,
        cost=cost_var,
        max_epoch=100,
    )

def main(argv=None):
    with tf.Graph().as_default():
        from train import prepare, start_train
        prepare()
        config = get_config()
        start_train(config)

if __name__ == '__main__':
    tf.app.run()
