#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10-preact18-mixup.py
# Author: Tao Hu <taohu620@gmail.com>,  Yauheni Selivonchyk <y.selivonchyk@gmail.com>

import numpy as np
import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset

BATCH_SIZE = 128
CLASS_NUM = 10

LR_SCHEDULE = [(0, 0.1), (100, 0.01), (150, 0.001)]
WEIGHT_DECAY = 1e-4

FILTER_SIZES = [64, 128, 256, 512]
MODULE_SIZES = [2, 2, 2, 2]


def preactivation_block(input, num_filters, stride=1):
    num_filters_in = input.get_shape().as_list()[1]

    # residual
    net = BNReLU(input)
    residual = Conv2D('conv1', net, num_filters, kernel_shape=3, stride=stride, use_bias=False, nl=BNReLU)
    residual = Conv2D('conv2', residual, num_filters, kernel_shape=3, stride=1, use_bias=False, nl=tf.identity)

    # identity
    shortcut = input
    if stride != 1 or num_filters_in != num_filters:
        shortcut = Conv2D('shortcut', net, num_filters, kernel_shape=1, stride=stride, use_bias=False,
                          nl=tf.identity)

    return shortcut + residual


class ResNet_Cifar(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.float32, [None, CLASS_NUM], 'label')]

    def _build_graph(self, inputs):
        assert tf.test.is_gpu_available()
        image, label = inputs

        MEAN_IMAGE = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        STD_IMAGE = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
        image = ((image / 255.0) - MEAN_IMAGE) / STD_IMAGE
        image = tf.transpose(image, [0, 3, 1, 2])

        pytorch_default_init = tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform')
        with argscope([Conv2D, BatchNorm, GlobalAvgPooling], data_format='NCHW'), \
                argscope(Conv2D, W_init=pytorch_default_init):
            net = Conv2D('conv0', image, 64, kernel_shape=3, stride=1, use_bias=False)
            for i, blocks_in_module in enumerate(MODULE_SIZES):
                for j in range(blocks_in_module):
                    stride = 2 if j == 0 and i > 0 else 1
                    with tf.variable_scope("res%d.%d" % (i, j)):
                        net = preactivation_block(net, FILTER_SIZES[i], stride)
            net = GlobalAvgPooling('gap', net)
            logits = FullyConnected('linear', net, out_dim=CLASS_NUM,
                                    nl=tf.identity, W_init=tf.random_normal_initializer(stddev=1e-3))

        ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
        ce_cost = tf.reduce_mean(ce_cost, name='cross_entropy_loss')

        single_label = tf.to_int32(tf.argmax(label, axis=1))
        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, single_label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'), ce_cost)
        add_param_summary(('.*/W', ['histogram']))

        # weight decay on all W matrixes. including convolutional layers
        wd_cost = tf.multiply(WEIGHT_DECAY, regularize_cost('.*', tf.nn.l2_loss), name='wd_cost')

        self.cost = tf.add_n([ce_cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test, isMixup, alpha):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
        ]
        ds = AugmentImageComponent(ds, augmentors)

    batch = BATCH_SIZE
    ds = BatchData(ds, batch, remainder=not isTrain)

    def f(dp):
        images, labels = dp
        one_hot_labels = np.eye(CLASS_NUM)[labels]  # one hot coding
        if not isTrain or not isMixup:
            return [images, one_hot_labels]

        # mixup:
        weight = np.random.beta(alpha, alpha, BATCH_SIZE)
        x_weight = weight.reshape(BATCH_SIZE, 1, 1, 1)
        y_weight = weight.reshape(BATCH_SIZE, 1)
        index = np.random.permutation(BATCH_SIZE)

        x1, x2 = images, images[index]
        x = x1 * x_weight + x2 * (1 - x_weight)
        y1, y2 = one_hot_labels, one_hot_labels[index]
        y = y1 * y_weight + y2 * (1 - y_weight)
        return [x, y]

    ds = MapData(ds, f)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--mixup', help='enable mixup', action='store_true')
    parser.add_argument('--alpha', default=1, type=float, help='alpha in mixup')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    log_foder = 'train_log/cifar10-preact18%s' % ('-mixup' if args.mixup else '')
    logger.set_logger_dir(os.path.join(log_foder))

    dataset_train = get_data('train', args.mixup, args.alpha)
    dataset_test = get_data('test', args.mixup, args.alpha)

    steps_per_epoch = dataset_train.size()

    config = TrainConfig(
        model=ResNet_Cifar(),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate', LR_SCHEDULE)
        ],
        max_epoch=200,
        steps_per_epoch=steps_per_epoch,
        session_init=SaverRestore(args.load) if args.load else None
    )
    launch_train_with_config(config, SimpleTrainer())
