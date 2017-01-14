#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn-digit-convnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
A very small SVHN convnet model (only 0.8m parameters).
About 2.3% validation error after 70 epochs. 2.15% after 150 epochs.

Each epoch iterates over the whole training set (4721 iterations).
Speed is about 43 it/s on TitanX.
"""


class Model(ModelDesc):

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 40, 40, 3], 'input'),
                InputVar(tf.int32, [None], 'label')]

    def _build_graph(self, input_vars):
        image, label = input_vars

        image = image / 128.0 - 1

        with argscope(Conv2D, nl=BNReLU, use_bias=False):
            logits = (LinearWrap(image)
                      .Conv2D('conv1', 24, 5, padding='VALID')
                      .MaxPooling('pool1', 2, padding='SAME')
                      .Conv2D('conv2', 32, 3, padding='VALID')
                      .Conv2D('conv3', 32, 3, padding='VALID')
                      .MaxPooling('pool2', 2, padding='SAME')
                      .Conv2D('conv4', 64, 3, padding='VALID')
                      .Dropout('drop', 0.5)
                      .FullyConnected('fc0', 512,
                                      b_init=tf.constant_initializer(0.1), nl=tf.nn.relu)
                      .FullyConnected('linear', out_dim=10, nl=tf.identity)())
        prob = tf.nn.softmax(logits, name='output')

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wd_cost = regularize_cost('fc.*/W', l2_regularizer(0.00001))
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram', 'rms']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')


def get_data():
    d1 = dataset.SVHNDigit('train')
    d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1, d2])
    data_test = dataset.SVHNDigit('test', shuffle=False)

    augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5, 1.5)),
        imgaug.GaussianDeform(  # this is slow. only use it when you have lots of cpus
            [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
            (40, 40), 0.2, 3),
    ]
    data_train = AugmentImageComponent(data_train, augmentors)
    data_train = BatchData(data_train, 128)
    data_train = PrefetchData(data_train, 5, 5)

    augmentors = [imgaug.Resize((40, 40))]
    data_test = AugmentImageComponent(data_test, augmentors)
    data_test = BatchData(data_test, 128, remainder=True)
    return data_train, data_test


def get_config():
    logger.auto_set_dir()

    data_train, data_test = get_data()
    step_per_epoch = data_train.size()

    lr = tf.train.exponential_decay(
        learning_rate=1e-3,
        global_step=get_global_step_var(),
        decay_steps=data_train.size() * 60,
        decay_rate=0.2, staircase=True, name='learning_rate')
    tf.summary.scalar('lr', lr)

    return TrainConfig(
        dataflow=data_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            InferenceRunner(data_test,
                            [ScalarStats('cost'), ClassificationError()])
        ]),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=350,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Graph().as_default():
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
