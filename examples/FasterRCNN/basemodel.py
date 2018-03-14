#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: basemodel.py

from contextlib import contextmanager
import tensorflow as tf
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.varreplace import custom_getter_scope
from tensorpack.models import (
    Conv2D, MaxPooling, BatchNorm, BNReLU)

import config


def maybe_freeze_affine(getter, *args, **kwargs):
    # custom getter to freeze affine params inside bn
    name = args[0] if len(args) else kwargs.get('name')
    if name.endswith('/gamma') or name.endswith('/beta'):
        if config.FREEZE_AFFINE:
            kwargs['trainable'] = False
    return getter(*args, **kwargs)


@contextmanager
def resnet_argscope():
    with argscope([Conv2D, MaxPooling, BatchNorm], data_format='channels_first'), \
            argscope(Conv2D, use_bias=False), \
            argscope(BatchNorm, training=False), \
            custom_getter_scope(maybe_freeze_affine):
        yield


def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)

        mean = [0.485, 0.456, 0.406]    # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std
        return image


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        if stride == 2:
            l = l[:, :, :-1, :-1]
            return Conv2D('convshortcut', l, n_out, 1,
                          strides=stride, padding='VALID', activation=activation)
        else:
            return Conv2D('convshortcut', l, n_out, 1,
                          strides=stride, activation=activation)
    else:
        return l


def resnet_bottleneck(l, ch_out, stride):
    l, shortcut = l, l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    if stride == 2:
        l = tf.pad(l, [[0, 0], [0, 0], [0, 1], [0, 1]])
        l = Conv2D('conv2', l, ch_out, 3, strides=2, activation=BNReLU, padding='VALID')
    else:
        l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features,
                               stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


def pretrained_resnet_conv4(image, num_blocks, freeze_c2=True):
    assert len(num_blocks) == 3
    with resnet_argscope():
        l = tf.pad(image, [[0, 0], [0, 0], [2, 3], [2, 3]])
        l = Conv2D('conv0', l, 64, 7, strides=2, activation=BNReLU, padding='VALID')
        l = tf.pad(l, [[0, 0], [0, 0], [0, 1], [0, 1]])
        l = MaxPooling('pool0', l, 3, strides=2, padding='VALID')
        c2 = resnet_group(l, 'group0', resnet_bottleneck, 64, num_blocks[0], 1)
        # TODO replace var by const to enable optimization
        if freeze_c2:
            c2 = tf.stop_gradient(c2)
        c3 = resnet_group(c2, 'group1', resnet_bottleneck, 128, num_blocks[1], 2)
        c4 = resnet_group(c3, 'group2', resnet_bottleneck, 256, num_blocks[2], 2)
    # 16x downsampling up to now
    return c4


@auto_reuse_variable_scope
def resnet_conv5(image, num_block):
    with resnet_argscope():
        l = resnet_group(image, 'group3', resnet_bottleneck, 512, num_block, 2)
        return l
