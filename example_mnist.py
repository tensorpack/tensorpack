#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: example_mnist.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import numpy as np
import os, sys
import argparse

from tensorpack.train import TrainConfig, start_train
from tensorpack.models import *
from tensorpack.utils import *
from tensorpack.utils.symbolic_functions import *
from tensorpack.utils.summary import *
from tensorpack.callbacks import *
from tensorpack.dataflow import *

BATCH_SIZE = 128
MIN_AFTER_DEQUEUE = 500
CAPACITY = MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE

def get_model(inputs, is_training):
    """
    Args:
        inputs: a list of input variable for training
        e.g.: [image_var, label_var] with:
            image_var: bx28x28
            label_var: bx1 integer
        is_training: a python bool variable
    Returns:
        the cost to minimize. scalar variable
    """
    is_training = bool(is_training)
    keep_prob = tf.constant(0.5 if is_training else 1.0)

    image, label = inputs
    image = tf.expand_dims(image, 3)    # add a single channel

    l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
    l = MaxPooling('pool0', l, 2)
    l = Conv2D('conv1', l, out_channel=40, kernel_shape=3)
    l = MaxPooling('pool1', l, 2)

    l = FullyConnected('fc0', l, 1024)
    l = tf.nn.dropout(l, keep_prob)

    # fc will have activation summary by default. disable this for the output layer
    logits = FullyConnected('fc1', l, out_dim=10,
                         summary_activation=False, nl=tf.identity)
    prob = tf.nn.softmax(logits, name='prob')

    y = one_hot(label, 10)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    cost = tf.reduce_mean(cost, name='cross_entropy_loss')
    tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost)

    # compute the number of failed samples, for ValidationError to use at test time
    wrong = tf.not_equal(
        tf.cast(tf.argmax(prob, 1), tf.int32), label)
    wrong = tf.cast(wrong, tf.float32)
    nr_wrong = tf.reduce_sum(wrong, name='wrong')
    # monitor training error
    tf.add_to_collection(
        MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

    # weight decay on all W of fc layers
    wd_cost = tf.mul(1e-4,
                     regularize_cost('fc.*/W', tf.nn.l2_loss),
                     name='regularize_loss')
    tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)

    add_param_summary('.*/W')   # monitor histogram of all W
    return tf.add_n([wd_cost, cost], name='cost')

def get_config():
    basename = os.path.basename(__file__)
    log_dir = os.path.join('train_log', basename[:basename.rfind('.')])
    logger.set_logger_file(os.path.join(log_dir, 'training.log'))

    IMAGE_SIZE = 28

    dataset_train = BatchData(dataset.Mnist('train'), 128)
    dataset_test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    step_per_epoch = dataset_train.size()
    #step_per_epoch = 3
    #dataset_test = FixedSizeData(dataset_test, 20)

    sess_config = get_default_sess_config()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # prepare model
    input_vars = [
        tf.placeholder(
            tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE), name='input'),
        tf.placeholder(
            tf.int32, shape=(None,), name='label')
    ]
    input_queue = tf.FIFOQueue(
        100, [x.dtype for x in input_vars], name='queue')

    lr = tf.train.exponential_decay(
        learning_rate=1e-4,
        global_step=get_global_step_var(),
        decay_steps=dataset_train.size() * 50,
        decay_rate=0.1, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            SummaryWriter(print_tag=['train_cost', 'train_error']),
            PeriodicSaver(),
            ValidationError(dataset_test, prefix='test'),
        ]),
        session_config=sess_config,
        inputs=input_vars,
        input_queue=input_queue,
        get_model_func=get_model,
        step_per_epoch=step_per_epoch,
        max_epoch=100,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with tf.Graph().as_default():
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        start_train(config)
