#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: example_svhn_digit.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack.train import TrainConfig, QueueInputTrainer
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.utils import *
from tensorpack.utils.symbolic_functions import *
from tensorpack.utils.summary import *
from tensorpack.dataflow import *
from tensorpack.dataflow import imgaug


class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 40, 40, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _get_cost(self, input_vars, is_training):
        image, label = input_vars
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        image = image / 255.0

        nl = lambda x, name: tf.abs(tf.tanh(x), name=name)
        l = Conv2D('conv1', image, 24, 5, padding='VALID', nl=nl)
        l = MaxPooling('pool1', l, 2, padding='SAME')
        l = Conv2D('conv2', l, 32, 3, nl=nl, padding='VALID')
        l = Conv2D('conv3', l, 32, 3, nl=nl, padding='VALID')
        l = MaxPooling('pool2', l, 2, padding='SAME')
        l = Conv2D('conv4', l, 64, 3, nl=nl, padding='VALID')

        l = tf.nn.dropout(l, keep_prob)
        l = FullyConnected('fc0', l, 512,
                           b_init=tf.constant_initializer(0.1), nl=nl)
        # fc will have activation summary by default. disable for the output layer
        logits = FullyConnected('linear', l, out_dim=10, summary_activation=False,
                                nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')

        y = one_hot(label, 10)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost)

        # compute the number of failed samples, for ValidationError to use at test time
        wrong = prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        tf.add_to_collection(
            MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.00001,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)

        add_param_summary([('.*/W', ['histogram', 'sparsity'])])   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

def get_config():
    #anchors = np.mgrid[0:4,0:4][:,1:,1:].transpose(1,2,0).reshape((-1,2)) / 4.0
    # prepare dataset
    d1 = dataset.SVHNDigit('train')
    d2 = dataset.SVHNDigit('extra')
    train = RandomMixData([d1, d2])
    test = dataset.SVHNDigit('test')

    augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.BrightnessAdd(63),
        imgaug.Contrast((0.2,1.8)),
    ]
    train = AugmentImageComponent(train, augmentors)
    train = BatchData(train, 128)
    nr_proc = 2
    train = PrefetchData(train, 3, nr_proc)
    step_per_epoch = train.size() / nr_proc

    augmentors = [
        imgaug.Resize((40, 40)),
    ]
    test = AugmentImageComponent(test, augmentors)
    test = BatchData(test, 128, remainder=True)

    sess_config = get_default_sess_config()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    lr = tf.train.exponential_decay(
        learning_rate=1e-4,
        global_step=get_global_step_var(),
        decay_steps=train.size() * 50,
        decay_rate=0.7, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(),
            PeriodicSaver(),
            ValidationError(test, prefix='test'),
        ]),
        session_config=sess_config,
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=100,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    basename = os.path.basename(__file__)
    logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]))

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
        QueueInputTrainer(config).train()
