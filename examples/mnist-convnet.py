#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf
import os, sys
import argparse

from tensorpack import *

"""
MNIST ConvNet example.
about 0.6% validation error after 30 epochs.
"""

IMAGE_SIZE = 28

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputVar(tf.int32, (None,), 'label') ]

    def _build_graph(self, input_vars):
        is_training = get_current_tower_context().is_training
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        image, label = input_vars
        image = tf.expand_dims(image, 3)    # add a single channel

        nl = PReLU.f
        image = image * 2 - 1
        with argscope(Conv2D, kernel_shape=3, nl=nl, out_channel=32):
            logits = (LinearWrap(image) # the starting brace is only for line-breaking
                    .Conv2D('conv0', padding='VALID')
                    .MaxPooling('pool0', 2)
                    .Conv2D('conv1', padding='SAME')
                    .Conv2D('conv2', padding='VALID')
                    .MaxPooling('pool1', 2)
                    .Conv2D('conv3', padding='VALID')
                    .FullyConnected('fc0', 512)
                    .tf.nn.dropout(keep_prob)
                    .FullyConnected('fc1', out_dim=10, nl=tf.identity)())
        prob = tf.nn.softmax(logits, name='prob')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = symbolic_functions.prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        summary.add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(1e-5,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        summary.add_moving_summary(cost, wd_cost)

        summary.add_param_summary([('.*/W', ['histogram'])])   # monitor histogram of all W
        self.cost = tf.add_n([wd_cost, cost], name='cost')

def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return train, test

def get_config():
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data()
    step_per_epoch = dataset_train.size()

    lr = tf.train.exponential_decay(
        learning_rate=1e-3,
        global_step=get_global_step_var(),
        decay_steps=dataset_train.size() * 10,
        decay_rate=0.3, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(dataset_test,
                [ScalarStats('cost'), ClassificationError() ]),
        ]),
        session_config=get_default_sess_config(0.5),
        model=Model(),
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

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    #QueueInputTrainer(config).train()
    SimpleTrainer(config).train()

