#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: example_cifar10.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack.models import *
from tensorpack.utils import *
from tensorpack.utils.symbolic_functions import *
from tensorpack.utils.summary import *
from tensorpack.utils.callback import *
from tensorpack.utils.validation_callback import *
from tensorpack.dataflow.dataset import Cifar10
from tensorpack.dataflow import *

BATCH_SIZE = 128
MIN_AFTER_DEQUEUE = 500
CAPACITY = MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE

def get_model(inputs, is_training):
    is_training = bool(is_training)
    keep_prob = tf.constant(0.5 if is_training else 1.0)

    image, label = inputs

    #if is_training:    # slow
        ## augmentations
        #image, label = tf.train.slice_input_producer(
            #[image, label], name='slice_queue')
        #image = tf.image.random_brightness(image, 0.1)
        #image, label = tf.train.shuffle_batch(
            #[image, label], BATCH_SIZE, CAPACITY, MIN_AFTER_DEQUEUE,
            #num_threads=2, enqueue_many=False)

    conv0 = Conv2D('conv0', image, out_channel=64, kernel_shape=5, padding='SAME')
    pool0 = MaxPooling('pool0', conv0, 3, stride=2, padding='SAME')
    norm0 = tf.nn.lrn(pool0, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm0')

    conv1 = Conv2D('conv1', norm0, out_channel=64, kernel_shape=5, padding='SAME')
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = MaxPooling('pool1', norm1, 3, stride=2, padding='SAME')

    fc0 = FullyConnected('fc0', pool1, 384)
    fc1 = FullyConnected('fc1', fc0, out_dim=192)
    # fc will have activation summary by default. disable this for the output layer
    fc2 = FullyConnected('fc2', fc1, out_dim=10, summary_activation=False, nl=tf.identity)
    prob = tf.nn.softmax(fc2, name='output')

    y = one_hot(label, 10)
    cost = tf.nn.softmax_cross_entropy_with_logits(fc2, y)
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
    wd_cost = tf.mul(1e-4,
                     regularize_cost('fc.*/W', tf.nn.l2_loss),
                     name='regularize_loss')
    tf.add_to_collection(COST_VARS_KEY, wd_cost)

    add_histogram_summary('.*/W')   # monitor histogram of all W
    # this won't work with multigpu
    #return [prob, nr_wrong], tf.add_n(tf.get_collection(COST_VARS_KEY), name='cost')
    return [prob, nr_wrong], tf.add_n([wd_cost, cost], name='cost')

def get_config():
    log_dir = os.path.join('train_log', os.path.basename(__file__)[:-3])
    logger.set_logger_dir(log_dir)

    import cv2
    dataset_train = Cifar10('train')
    dataset_train = MapData(dataset_train, lambda img: cv2.resize(img, (24, 24)))
    dataset_train = BatchData(dataset_train, 128)
    #dataset_test = BatchData(Cifar10('test'), 128)
    step_per_epoch = dataset_train.size()
    #step_per_epoch = 20
    #dataset_test = FixedSizeData(dataset_test, 20)

    sess_config = tf.ConfigProto()
    sess_config.device_count['GPU'] = 1
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess_config.gpu_options.allocator_type = 'BFC'
    sess_config.allow_soft_placement = True

    # prepare model
    input_vars = [
        tf.placeholder(
            tf.float32, shape=(None, 24, 24, 3), name='input'),
        tf.placeholder(
            tf.int32, shape=(None,), name='label')
    ]
    input_queue = tf.RandomShuffleQueue(
        100, 50, [x.dtype for x in input_vars], name='queue')

    global_step_var = tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    lr = tf.train.exponential_decay(
        learning_rate=1e-4,
        global_step=global_step_var,
        decay_steps=dataset_train.size() * 50,
        decay_rate=0.1, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return dict(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callback=Callbacks([
            SummaryWriter(),
            PeriodicSaver(),
            #ValidationError(dataset_test, prefix='test'),
        ]),
        session_config=sess_config,
        inputs=input_vars,
        input_queue=input_queue,
        get_model_func=get_model,
        step_per_epoch=step_per_epoch,
        max_epoch=100,
    )

if __name__ == '__main__':
    from tensorpack import train
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with tf.Graph().as_default():
        train.prepare()
        config = get_config()
        train.start_train(config)
