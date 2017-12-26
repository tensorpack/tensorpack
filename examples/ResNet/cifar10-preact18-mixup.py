#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10-preact18-mixup.py
# Author: Tao Hu <taohu620@gmail.com>

import numpy as np
import argparse
import os


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow import dataset

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

"""
This implementation uses the architecture of PreAct in:
https://github.com/kuangliu/pytorch-cifar

The implementation as used by the "mixup: Beyond Empirical Risk Minimization" paper https://arxiv.org/pdf/1710.09412.pdf

Results:
Test error with the original 100-150-200 schedule (validation set used as part of training set):
no mixup - 5.7%; mixup(alpha=1) - 4.1% (mixup paper: 5.6%/3.8%)

Results:
Validation error with the original 100-150-200 schedule on ResNet-18:
wd=0.0005: 5.36%/4.45% (without/with mixup)
wd=0.0001: 5,78%/4.16% (without/with mixup)

Usage:
./cifar10-preact18-mixup.py                     # train preactivation resnet18
./cifar10-preact18-mixup.py --depth=50          # train preactivation resnet50 with bottleneck
./cifar10-preact18-mixup.py --mixup             # apply mixup regularization
./cifar10-preact18-mixup.py --mixup --alpha=0.7 # apply mixup regularization with custom alpha
"""

BATCH_SIZE = 128
CLASS_NUM = 10


# Reference network hyperparameters as implemented by K. Liu, 2017. URL https://github.com/kuangliu/pytorch-cifar.
# which is

LR_SCHEDULE = [(1, 0.1), (100, 0.01), (150, 0.001)]
WEIGHT_DECAY = 0.0001

FILTER_SIZES = [64, 128, 256, 512]


def preactivation_block(input, num_filters, stride=1):
    num_filters_in = input.get_shape().as_list()[1]
    # identity
    shortcut = input
    if stride != 1 or num_filters_in != num_filters:
        shortcut = Conv2D('shortcut', input, num_filters, kernel_shape=1, stride=stride, use_bias=False,
                          nl=tf.identity)
    # residual
    residual = BNReLU(input)
    residual = Conv2D('conv1', residual, num_filters, kernel_shape=3, stride=stride, use_bias=False, nl=BNReLU)
    residual = Conv2D('conv2', residual, num_filters, kernel_shape=3, stride=1, use_bias=False, nl=tf.identity)
    return shortcut + residual


def bottleneck_block(input, num_filters, stride=1):
    expansion = 4

    num_filters_in = net.get_shape().as_list()[1]
    # identity
    shortcut = input
    if stride != 1 or num_filters_in != num_filters * expansion:
        shortcut = Conv2D('shortcut', input, num_filters * expansion, kernel_shape=1, stride=stride, use_bias=False,
                          nl=tf.identity)
    # residual
    res = BNReLU(input)
    res = Conv2D('conv1', res, num_filters, kernel_shape=1, stride=1, use_bias=False, nl=BNReLU)
    res = Conv2D('conv2', res, num_filters, kernel_shape=3, stride=stride, use_bias=False, nl=BNReLU)
    res = Conv2D('conv3', res, num_filters * expansion, kernel_shape=1, stride=1, use_bias=False, nl=tf.identity)
    return shortcut + res


RESNET_CONFIG = {
    18: {'block_func': preactivation_block, 'modules': [2, 2, 2, 2]},
    34: {'block_func': preactivation_block, 'modules': [3, 4, 6, 3]},
    50: {'block_func': bottleneck_block, 'modules': [3, 4, 6, 3]},
    101: {'block_func': bottleneck_block, 'modules': [3, 4, 23, 3]},
    152: {'block_func': bottleneck_block, 'modules': [3, 8, 36, 3]},
}


class ResNet_Cifar(ModelDesc):
    # module configuration taken from reference implementation by kuangliu.github.com
    def __init__(self, depth=18):
        super(ResNet_Cifar, self).__init__()

        if depth not in RESNET_CONFIG:
            print('Could not find configuration for depth "%d". Try one of %s' % (depth, RESNET_CONFIG.keys()))

        self.depth = depth

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.float32, [None, CLASS_NUM], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        config = RESNET_CONFIG[self.depth]
        block_function = config['block_func']
        module_sizes = config['modules']

        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='NCHW'), \
                argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3,
                         W_init=variance_scaling_initializer(mode='FAN_OUT')):
            net = Conv2D('conv0', image, FILTER_SIZES[0], kernel_shape=3, stride=1, use_bias=False)
            for i, blocks_in_module in enumerate(module_sizes):
                for j in range(blocks_in_module):
                    stride = 2 if j == 0 and i > 0 else 1
                    with tf.variable_scope("res%d.%d" % (i, j)):
                        net = block_function(net, FILTER_SIZES[i], stride)
            net = GlobalAvgPooling('gap', net)
        logits = FullyConnected('linear', net, out_dim=CLASS_NUM, nl=tf.identity)

        ce_cost = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits)
        ce_cost = tf.reduce_mean(ce_cost, name='cross_entropy_loss')

        single_label = tf.to_int32(tf.argmax(label, axis=1))
        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, single_label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W matrixes. including convolutional layers
        wd_cost = tf.multiply(WEIGHT_DECAY, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_param_summary(('.*/W', ['histogram']))   # monitor W

        self.cost = tf.add_n([ce_cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test, isMixup, alpha):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)

    if isMixup:
        batch = 2 * BATCH_SIZE
    else:
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
        x1, x2 = np.split(images, 2, axis=0)
        x = x1 * x_weight + x2 * (1 - x_weight)
        y1, y2 = np.split(one_hot_labels, 2, axis=0)
        y = y1 * y_weight + y2 * (1 - y_weight)
        return [x, y]

    ds = MapData(ds, f)

    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--depth', default=18, type=int, help='model depth. one of [18, 34, 50, 101, 152]')
    parser.add_argument('--mixup', help='enable mixup', action='store_true')
    parser.add_argument('--alpha', default=1, type=float, help='alpha in mixup')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    log_foder = 'train_log/cifar10-preact%d%s' % (args.depth, '-mixup' if args.mixup else '')
    logger.set_logger_dir(os.path.join(log_foder))

    dataset_train = get_data('train', args.mixup, args.alpha)
    dataset_test = get_data('test', args.mixup, args.alpha)

    steps_per_epoch = dataset_train.size()
    # because mixup utilize two data to generate one data, so the learning rate schedule are doubled.
    if args.mixup:
        steps_per_epoch *= 2

    config = TrainConfig(
        model=ResNet_Cifar(),
        dataflow=dataset_train,
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
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
