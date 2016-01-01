#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: example_cifar10.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack.train import TrainConfig, start_train
from tensorpack.models import *
from tensorpack.utils import *
from tensorpack.utils.symbolic_functions import *
from tensorpack.utils.summary import *
from tensorpack.callbacks import *
from tensorpack.dataflow import *
from tensorpack.dataflow import imgaug

BATCH_SIZE = 128
MIN_AFTER_DEQUEUE = 20000   # a large number, as in the official example
CAPACITY = MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE

def get_model(inputs, is_training):
    is_training = bool(is_training)
    keep_prob = tf.constant(0.5 if is_training else 1.0)

    image, label = inputs

    if is_training:
        image, label = tf.train.shuffle_batch(
            [image, label], BATCH_SIZE, CAPACITY, MIN_AFTER_DEQUEUE,
            num_threads=6, enqueue_many=False)
        ## augmentations
        #image, label = tf.train.slice_input_producer(
            #[image, label], name='slice_queue')
        #image = tf.image.random_brightness(image, 0.1)
        #image, label = tf.train.shuffle_batch(
            #[image, label], BATCH_SIZE, CAPACITY, MIN_AFTER_DEQUEUE,
            #num_threads=2, enqueue_many=False)

    l = Conv2D('conv0', image, out_channel=64, kernel_shape=5, padding='SAME')
    l = MaxPooling('pool0', l, 3, stride=2, padding='SAME')
    l = tf.nn.lrn(l, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm0')

    l = Conv2D('conv1', l, out_channel=64, kernel_shape=5, padding='SAME',
               b_init=tf.constant_initializer(0.1))
    l = tf.nn.lrn(l, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    l = MaxPooling('pool1', l, 3, stride=2, padding='SAME')

    l = FullyConnected('fc0', l, 384,
                       b_init=tf.constant_initializer(0.1))
    l = FullyConnected('fc1', l, out_dim=192,
                       b_init=tf.constant_initializer(0.1))
    # fc will have activation summary by default. disable this for the output layer
    logits = FullyConnected('linear', l, out_dim=10, summary_activation=False,
                            nl=tf.identity,
                            W_init=tf.truncated_normal_initializer(1/192.0))
    prob = tf.nn.softmax(logits, name='output')

    y = one_hot(label, 10)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    cost = tf.reduce_mean(cost, name='cross_entropy_loss')
    tf.add_to_collection(COST_VARS_KEY, cost)

    # compute the number of failed samples, for ValidationError to use at test time
    wrong = tf.not_equal(
        tf.cast(tf.argmax(prob, 1), tf.int32), label)
    wrong = tf.cast(wrong, tf.float32)
    nr_wrong = tf.reduce_sum(wrong, name='wrong')
    # monitor training error
    tf.add_to_collection(
        SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

    # weight decay on all W of fc layers
    wd_cost = tf.mul(4e-3,
                     regularize_cost('fc.*/W', tf.nn.l2_loss),
                     name='regularize_loss')
    tf.add_to_collection(COST_VARS_KEY, wd_cost)

    add_histogram_summary('.*/W')   # monitor histogram of all W
    # this won't work with multigpu
    #return [prob, nr_wrong], tf.add_n(tf.get_collection(COST_VARS_KEY), name='cost')
    return [prob, nr_wrong], tf.add_n([wd_cost, cost], name='cost')

def get_config():
    basename = os.path.basename(__file__)
    log_dir = os.path.join('train_log', basename[:basename.rfind('.')])
    logger.set_logger_dir(log_dir)

    dataset_train = dataset.Cifar10('train')
    augmentors = [
        RandomCrop((24, 24)),
        Flip(horiz=True),
        BrightnessAdd(0.25),
        Contrast((0.2,1.8)),
        PerImageWhitening()
    ]
    dataset_train = AugmentImageComponent(dataset_train, augmentors)
    dataset_train = BatchData(dataset_train, 128)

    augmentors = [
        CenterCrop((24, 24)),
        PerImageWhitening()
    ]
    dataset_test = dataset.Cifar10('test')
    dataset_test = AugmentImageComponent(dataset_test, augmentors)
    dataset_test = BatchData(dataset_test, 128)
    step_per_epoch = dataset_train.size()
    #step_per_epoch = 20
    #dataset_test = FixedSizeData(dataset_test, 20)

    sess_config = get_default_sess_config()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # prepare model
    input_vars = [
        tf.placeholder(
            tf.float32, shape=(None, 24, 24, 3), name='input'),
        tf.placeholder(
            tf.int32, shape=(None,), name='label')
    ]
    input_queue = tf.FIFOQueue(
        50, [x.dtype for x in input_vars], name='queue')

    lr = tf.train.exponential_decay(
        learning_rate=1e-1,
        global_step=get_global_step_var(),
        decay_steps=dataset_train.size() * 350,
        decay_rate=0.1, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.GradientDescentOptimizer(lr),
        callbacks=Callbacks([
            SummaryWriter(print_tag=['train_cost', 'train_error']),
            PeriodicSaver(),
            ValidationError(dataset_test, prefix='test'),
        ]),
        session_config=sess_config,
        inputs=input_vars,
        input_queue=input_queue,
        get_model_func=get_model,
        batched_model_input=False,
        step_per_epoch=step_per_epoch,
        max_epoch=500,
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
