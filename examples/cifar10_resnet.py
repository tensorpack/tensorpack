#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10_resnet.py
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

"""
CIFAR10-resnet example.
I can reproduce the results in:
Deep Residual Learning for Image Recognition, arxiv:1512.03385
for n=5 and 18
This model achieves slightly better results due to the use of the
whole training set instead of a 95:5 train-val split.
"""

BATCH_SIZE = 128

class Model(ModelDesc):
    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 32, 32, 3], 'input'),
                InputVar(tf.int32, [None], 'label')
               ]

    def _get_cost(self, input_vars, is_training):
        image, label = input_vars
        image = image / 255.0

        def conv(name, l, channel, stride):
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.contrib.layers.xavier_initializer_conv2d(False))

        def residual(name, l, increase_dim=False):
            shape = l.get_shape().as_list()
            in_channel = shape[3]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name) as scope:
                c1 = conv('conv1', l, out_channel, stride1)
                b1 = BatchNorm('bn1', c1, is_training)
                b1 = tf.nn.relu(b1)
                c2 = conv('conv2', b1, out_channel, 1)
                b2 = BatchNorm('bn2', c2, is_training)

                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0,0], [0,0], [0,0], [in_channel//2, in_channel//2]])

                l = b2 + l
                l = tf.nn.relu(l)
                return l


        l = conv('conv1', image, 16, 1)
        l = BatchNorm('bn1', l, is_training)
        l = tf.nn.relu(l)
        for k in range(self.n):
            l = residual('res1.{}'.format(k), l)
        # 32,c=16

        l = residual('res2.0', l, increase_dim=True)
        for k in range(1, self.n):
            l = residual('res2.{}'.format(k), l)
        # 16,c=32

        l = residual('res3.0', l, increase_dim=True)
        for k in range(1, self.n):
            l = residual('res3.' + str(k), l)
        # 8,c=64
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, out_dim=10, summary_activation=False,
                                nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')

        y = one_hot(label, 10)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost)

        wrong = prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        tf.add_to_collection(
            MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('.*/W', l2_regularizer(0.0002), name='regularize_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)

        add_param_summary([('.*/W', ['histogram', 'sparsity'])])   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.BrightnessAdd(20),
            imgaug.Contrast((0.6,1.4)),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds



def get_config():
    # prepare dataset
    dataset_train = get_data('train')
    step_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    sess_config = get_default_sess_config(0.9)

    lr = tf.train.exponential_decay(
        learning_rate=1e-1,
        global_step=get_global_step_var(),
        decay_steps=36000,
        decay_rate=0.1, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=Callbacks([
            StatPrinter(),
            PeriodicSaver(),
            ValidationError(dataset_test, prefix='test'),
        ]),
        session_config=sess_config,
        model=Model(n=18),
        step_per_epoch=step_per_epoch,
        max_epoch=500,
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

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        QueueInputTrainer(config).train()
