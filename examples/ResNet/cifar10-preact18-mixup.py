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
This is different from the one in cifar10-resnet.py

Results:
Validation error with the original 100-150-200 schedule:
no mixup - 5.0%; mixup(alpha=1) - 3.8%

Using 2x learning schedule, it can further improve to 4.7% and 3.2%.

Usage:
./cifar10-preact18-mixup.py  # train without mixup
./cifar10-preact18-mixup.py --mixup	 # with mixup
"""

BATCH_SIZE = 128
CLASS_NUM = 10


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.float32, [None, CLASS_NUM], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        def preactblock(input, name, in_planes, planes, stride=1):
            with tf.variable_scope(name):
                input2 = BNReLU(input)
                if stride != 1 or in_planes != planes:
                    shortcut = Conv2D('shortcut', input2, planes, kernel_shape=1, stride=stride, use_bias=False,
                                      nl=tf.identity)
                else:
                    shortcut = input
                input2 = Conv2D('conv1', input2, planes, kernel_shape=3, stride=1, use_bias=False, nl=BNReLU)
                input2 = Conv2D('conv2', input2, planes, kernel_shape=3, stride=stride, use_bias=False, nl=BNReLU)
                input2 = Conv2D('conv3', input2, planes, kernel_shape=3, stride=1, use_bias=False, nl=tf.identity)
                input2 += shortcut
            return input2

        def _make_layer(input, planes, num_blocks, current_plane, stride, name):
            strides = [stride] + [1] * (num_blocks - 1)  # first block stride = stride, the latter block stride = 1
            for index, stride in enumerate(strides):
                input = preactblock(input, "{}.{}".format(name, index), current_plane, planes, stride)
                current_plane = planes
            return input, current_plane

        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='NCHW'), \
                argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3,
                         W_init=variance_scaling_initializer(mode='FAN_OUT')):

            l = Conv2D('conv0', image, 64, kernel_shape=3, stride=1, use_bias=False)

            current_plane = 64
            l, current_plane = _make_layer(l, 64, 2, current_plane, stride=1, name="res1")
            l, current_plane = _make_layer(l, 128, 2, current_plane, stride=2, name="res2")
            l, current_plane = _make_layer(l, 256, 2, current_plane, stride=2, name="res3")
            l, current_plane = _make_layer(l, 512, 2, current_plane, stride=2, name="res4")
            l = GlobalAvgPooling('gap', l)

        logits = FullyConnected('linear', l, out_dim=CLASS_NUM, nl=tf.identity)

        cost = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        single_label = tf.to_int32(tf.argmax(label, axis=1))
        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, single_label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

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

    def f(ds):
        images, labels = ds
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
    parser.add_argument('--mixup', help='enable mixup', action='store_true')
    parser.add_argument('--alpha', default=1, type=float, help='alpha in mixup')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_logger_dir(
        os.path.join('train_log/cifar10-preact18-{}mixup'.format('' if args.mixup else 'no')))

    dataset_train = get_data('train', args.mixup, args.alpha)
    dataset_test = get_data('test', args.mixup, args.alpha)

    steps_per_epoch = dataset_train.size()
    # because mixup utilize two data to generate one data, so the learning rate schedule are doubled.
    if args.mixup:
        steps_per_epoch *= 2

    config = TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (100, 0.01), (150, 0.001)])
        ],
        max_epoch=200,
        steps_per_epoch=steps_per_epoch,
        session_init=SaverRestore(args.load) if args.load else None
    )
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
