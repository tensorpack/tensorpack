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
from tensorpack.callbacks import *
from tensorpack.utils import *
from tensorpack.utils.symbolic_functions import *
from tensorpack.utils.summary import *
from tensorpack.dataflow import *
from tensorpack.dataflow import imgaug

"""
CIFAR10 89% test accuracy after 200 epochs.
"""

BATCH_SIZE = 128
MIN_AFTER_DEQUEUE = int(50000 * 0.4)
CAPACITY = MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE

class Model(ModelDesc):
    def _get_input_vars(self):
        return [
            tf.placeholder(
                tf.float32, shape=[None, 30, 30, 3], name='input'),
            tf.placeholder(
                tf.int32, shape=[None], name='label')
        ]

    def _get_cost(self, input_vars, is_training):
        image, label = input_vars

        if is_training:
            image, label = tf.train.shuffle_batch(
                [image, label], BATCH_SIZE, CAPACITY, MIN_AFTER_DEQUEUE,
                num_threads=6, enqueue_many=True)
            tf.image_summary("train_image", image, 10)

        l = Conv2D('conv1.1', image, out_channel=64, kernel_shape=3, padding='SAME')
        l = Conv2D('conv1.2', l, out_channel=64, kernel_shape=3, nl=tf.identity)
        l = BatchNorm('bn1', l, is_training)
        l = tf.nn.relu(l)
        l = MaxPooling('pool1', l, 3, stride=2, padding='SAME')

        l = Conv2D('conv2.1', l, out_channel=128, kernel_shape=3)
        l = Conv2D('conv2.2', l, out_channel=128, kernel_shape=3, nl=tf.identity)
        l = BatchNorm('bn2', l, is_training)
        l = tf.nn.relu(l)
        l = MaxPooling('pool2', l, 3, stride=2, padding='SAME')

        l = Conv2D('conv3.1', l, out_channel=128, kernel_shape=3, padding='VALID')
        l = Conv2D('conv3.2', l, out_channel=128, kernel_shape=3, padding='VALID', nl=tf.identity)
        l = BatchNorm('bn3', l, is_training)
        l = tf.nn.relu(l)
        l = FullyConnected('fc0', l, 512,
                           W_init=tf.truncated_normal_initializer(stddev=0.04),
                           b_init=tf.constant_initializer(0.1))
        l = FullyConnected('fc1', l, out_dim=512,
                           W_init=tf.truncated_normal_initializer(stddev=0.04),
                           b_init=tf.constant_initializer(0.1))
        # fc will have activation summary by default. disable for the output layer
        logits = FullyConnected('linear', l, out_dim=10, summary_activation=False,
                                nl=tf.identity,
                                W_init=tf.truncated_normal_initializer(stddev=1.0/192))
        prob = tf.nn.softmax(logits, name='output')

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
        wd_cost = tf.mul(0.004,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)

        add_param_summary('.*')   # monitor all variables
        return tf.add_n([cost, wd_cost], name='cost')

def get_config():
    basename = os.path.basename(__file__)
    logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]))

    # prepare dataset
    dataset_train = dataset.Cifar10('train')
    augmentors = [
        imgaug.RandomCrop((30, 30)),
        imgaug.Flip(horiz=True),
        imgaug.BrightnessAdd(63),
        imgaug.Contrast((0.2,1.8)),
        imgaug.MeanVarianceNormalize(all_channel=True)
    ]
    dataset_train = AugmentImageComponent(dataset_train, augmentors)
    dataset_train = BatchData(dataset_train, 128)
    step_per_epoch = dataset_train.size()

    augmentors = [
        imgaug.CenterCrop((30, 30)),
        imgaug.MeanVarianceNormalize(all_channel=True)
    ]
    dataset_test = dataset.Cifar10('test')
    dataset_test = AugmentImageComponent(dataset_test, augmentors)
    dataset_test = BatchData(dataset_test, 128, remainder=True)

    sess_config = get_default_sess_config()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    lr = tf.train.exponential_decay(
        learning_rate=1e-2,
        global_step=get_global_step_var(),
        decay_steps=dataset_train.size() * 30,
        decay_rate=0.5, staircase=True, name='learning_rate')
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
        model=Model(),
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
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        start_train(config)
