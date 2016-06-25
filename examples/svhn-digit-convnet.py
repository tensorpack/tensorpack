#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn-digit-convnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
SVHN convnet.
About 3.0% validation error after 70 epoch.  2.5% after 130 epoch.
"""

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 40, 40, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars, is_training):
        image, label = input_vars
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        image = image / 128.0 - 1

        logits = LinearWrap(image) \
                .Conv2D('conv1', 24, 5, padding='VALID') \
                .MaxPooling('pool1', 2, padding='SAME') \
                .Conv2D('conv2', 32, 3, padding='VALID') \
                .Conv2D('conv3', 32, 3, padding='VALID') \
                .MaxPooling('pool2', 2, padding='SAME') \
                .Conv2D('conv4', 64, 3, padding='VALID') \
                .tf.nn.dropout(keep_prob) \
                .FullyConnected('fc0', 512,
                        b_init=tf.constant_initializer(0.1)) \
                .FullyConnected('linear', out_dim=10, nl=tf.identity)()
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost)

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        tf.add_to_collection(
            MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(0.00001))
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)

        add_param_summary([('.*/W', ['histogram', 'sparsity'])])   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

def get_config():
    # prepare dataset
    d1 = dataset.SVHNDigit('train')
    d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1, d2])
    data_test = dataset.SVHNDigit('test')

    augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5,1.5)),
        imgaug.GaussianDeform(  # this is slow. only use it when you have lots of cpus
            [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
            (40,40), 0.2, 3),
    ]
    data_train = AugmentImageComponent(data_train, augmentors)
    data_train = BatchData(data_train, 128)
    data_train = PrefetchData(data_train, 5, 5)
    step_per_epoch = data_train.size()

    augmentors = [ imgaug.Resize((40, 40)) ]
    data_test = AugmentImageComponent(data_test, augmentors)
    data_test = BatchData(data_test, 128, remainder=True)

    lr = tf.train.exponential_decay(
        learning_rate=1e-3,
        global_step=get_global_step_var(),
        decay_steps=data_train.size() * 60,
        decay_rate=0.2, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=data_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(data_test,
                [ScalarStats('cost'), ClassificationError()])
        ]),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=350,
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
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        QueueInputTrainer(config).train()
