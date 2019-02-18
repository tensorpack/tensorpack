# -*- coding: utf-8 -*-
# File: basemodel.py

import numpy as np
from contextlib import ExitStack, contextmanager
import tensorflow as tf

from tensorpack.models import BatchNorm, Conv2D, MaxPooling, layer_register
from tensorpack.tfutils import argscope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.varreplace import custom_getter_scope, freeze_variables

from config import config as cfg


@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=tf.constant_initializer(1.)):
    """
    More code that reproduces the paper can be found at https://github.com/ppwwyyxx/GroupNorm-reproduce/.
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')


def freeze_affine_getter(getter, *args, **kwargs):
    # custom getter to freeze affine params inside bn
    name = args[0] if len(args) else kwargs.get('name')
    if name.endswith('/gamma') or name.endswith('/beta'):
        kwargs['trainable'] = False
        ret = getter(*args, **kwargs)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, ret)
    else:
        ret = getter(*args, **kwargs)
    return ret


def maybe_reverse_pad(topleft, bottomright):
    if cfg.BACKBONE.TF_PAD_MODE:
        return [topleft, bottomright]
    return [bottomright, topleft]


@contextmanager
def backbone_scope(freeze):
    """
    Args:
        freeze (bool): whether to freeze all the variables under the scope
    """
    def nonlin(x):
        x = get_norm()(x)
        return tf.nn.relu(x)

    with argscope([Conv2D, MaxPooling, BatchNorm], data_format='channels_first'), \
            argscope(Conv2D, use_bias=False, activation=nonlin,
                     kernel_initializer=tf.variance_scaling_initializer(
                         scale=2.0, mode='fan_out')), \
            ExitStack() as stack:
        if cfg.BACKBONE.NORM in ['FreezeBN', 'SyncBN']:
            if freeze or cfg.BACKBONE.NORM == 'FreezeBN':
                stack.enter_context(argscope(BatchNorm, training=False))
            else:
                stack.enter_context(argscope(
                    BatchNorm, sync_statistics='nccl' if cfg.TRAINER == 'replicated' else 'horovod'))

        if freeze:
            stack.enter_context(freeze_variables(stop_gradient=False, skip_collection=True))
        else:
            # the layers are not completely freezed, but we may want to only freeze the affine
            if cfg.BACKBONE.FREEZE_AFFINE:
                stack.enter_context(custom_getter_scope(freeze_affine_getter))
        yield


def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.PREPROC.PIXEL_MEAN
        std = np.asarray(cfg.PREPROC.PIXEL_STD)
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean) * image_invstd
        return image


def get_norm(zero_init=False):
    if cfg.BACKBONE.NORM == 'None':
        return lambda x: x
    if cfg.BACKBONE.NORM == 'GN':
        Norm = GroupNorm
        layer_name = 'gn'
    else:
        Norm = BatchNorm
        layer_name = 'bn'
    return lambda x: Norm(layer_name, x, gamma_initializer=tf.zeros_initializer() if zero_init else None)


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.shape[1]
    if n_in != n_out:   # change dimension when channel is not the same
        # TF's SAME mode output ceil(x/stride), which is NOT what we want when x is odd and stride is 2
        # In FPN mode, the images are pre-padded already.
        if not cfg.MODE_FPN and stride == 2:
            l = l[:, :, :-1, :-1]
        return Conv2D('convshortcut', l, n_out, 1,
                      strides=stride, activation=activation)
    else:
        return l


def resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    if cfg.BACKBONE.STRIDE_1X1:
        if stride == 2:
            l = l[:, :, :-1, :-1]
        l = Conv2D('conv1', l, ch_out, 1, strides=stride)
        l = Conv2D('conv2', l, ch_out, 3, strides=1)
    else:
        l = Conv2D('conv1', l, ch_out, 1, strides=1)
        if stride == 2:
            l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1)])
            l = Conv2D('conv2', l, ch_out, 3, strides=2, padding='VALID')
        else:
            l = Conv2D('conv2', l, ch_out, 3, strides=stride)
    if cfg.BACKBONE.NORM != 'None':
        l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_norm(zero_init=True))
    else:
        l = Conv2D('conv3', l, ch_out * 4, 1, activation=tf.identity,
                   kernel_initializer=tf.constant_initializer())
    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_norm(zero_init=False))
    return tf.nn.relu(ret, name='output')


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_c4_backbone(image, num_blocks):
    assert len(num_blocks) == 3
    freeze_at = cfg.BACKBONE.FREEZE_AT
    with backbone_scope(freeze=freeze_at > 0):
        l = tf.pad(image, [[0, 0], [0, 0], maybe_reverse_pad(2, 3), maybe_reverse_pad(2, 3)])
        l = Conv2D('conv0', l, 64, 7, strides=2, padding='VALID')
        l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1)])
        l = MaxPooling('pool0', l, 3, strides=2, padding='VALID')

    with backbone_scope(freeze=freeze_at > 1):
        c2 = resnet_group('group0', l, resnet_bottleneck, 64, num_blocks[0], 1)
    with backbone_scope(freeze=False):
        c3 = resnet_group('group1', c2, resnet_bottleneck, 128, num_blocks[1], 2)
        c4 = resnet_group('group2', c3, resnet_bottleneck, 256, num_blocks[2], 2)
    # 16x downsampling up to now
    return c4


@auto_reuse_variable_scope
def resnet_conv5(image, num_block):
    with backbone_scope(freeze=False):
        l = resnet_group('group3', image, resnet_bottleneck, 512, num_block, 2)
        return l


def resnet_fpn_backbone(image, num_blocks):
    freeze_at = cfg.BACKBONE.FREEZE_AT
    shape2d = tf.shape(image)[2:]
    mult = float(cfg.FPN.RESOLUTION_REQUIREMENT)
    new_shape2d = tf.cast(tf.ceil(tf.cast(shape2d, tf.float32) / mult) * mult, tf.int32)
    pad_shape2d = new_shape2d - shape2d
    assert len(num_blocks) == 4, num_blocks
    with backbone_scope(freeze=freeze_at > 0):
        chan = image.shape[1]
        pad_base = maybe_reverse_pad(2, 3)
        l = tf.pad(image, tf.stack(
            [[0, 0], [0, 0],
             [pad_base[0], pad_base[1] + pad_shape2d[0]],
             [pad_base[0], pad_base[1] + pad_shape2d[1]]]))
        l.set_shape([None, chan, None, None])
        l = Conv2D('conv0', l, 64, 7, strides=2, padding='VALID')
        l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1)])
        l = MaxPooling('pool0', l, 3, strides=2, padding='VALID')
    with backbone_scope(freeze=freeze_at > 1):
        c2 = resnet_group('group0', l, resnet_bottleneck, 64, num_blocks[0], 1)
    with backbone_scope(freeze=False):
        c3 = resnet_group('group1', c2, resnet_bottleneck, 128, num_blocks[1], 2)
        c4 = resnet_group('group2', c3, resnet_bottleneck, 256, num_blocks[2], 2)
        c5 = resnet_group('group3', c4, resnet_bottleneck, 512, num_blocks[3], 2)
    # 32x downsampling up to now
    # size of c5: ceil(input/32)
    return c2, c3, c4, c5
