# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack import tfv1
from tensorpack.models import Conv2D, Conv2DTranspose, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary

from .backbone import GroupNorm
from config import config as cfg


@under_name_scope()
def maskrcnn_loss(mask_logits, fg_labels, fg_target_masks):
    """
    Args:
        mask_logits: #fg x #category xhxw
        fg_labels: #fg, in 1~#class, int64
        fg_target_masks: #fgxhxw, float32
    """
    mask_logits = tf.gather(
        mask_logits, tf.reshape(fg_labels - 1, [-1, 1]), batch_dims=1)
    mask_logits = tf.squeeze(mask_logits, axis=1)

    mask_probs = tf.sigmoid(mask_logits)

    # add some training visualizations to tensorboard
    with tf.name_scope('mask_viz'):
        viz = tf.concat([fg_target_masks, mask_probs], axis=1)
        viz = tf.expand_dims(viz, 3)
        viz = tf.cast(viz * 255, tf.uint8, name='viz')
        tf.summary.image('mask_truth|pred', viz, max_outputs=10)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=fg_target_masks, logits=mask_logits)
    loss = tf.reduce_mean(loss, name='maskrcnn_loss')

    pred_label = mask_probs > 0.5
    truth_label = fg_target_masks > 0.5
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(pred_label, truth_label), tf.float32),
        name='accuracy')
    pos_accuracy = tf.logical_and(
        tf.equal(pred_label, truth_label),
        tf.equal(truth_label, True))
    pos_accuracy = tf.reduce_mean(tf.cast(pos_accuracy, tf.float32), name='pos_accuracy')
    fg_pixel_ratio = tf.reduce_mean(tf.cast(truth_label, tf.float32), name='fg_pixel_ratio')

    add_moving_summary(loss, accuracy, fg_pixel_ratio, pos_accuracy)
    return loss


@layer_register(log_shape=True)
def maskrcnn_upXconv_head(feature, num_category, num_convs, norm=None):
    """
    Args:
        feature (NxCx s x s): size is 7 in C4 models and 14 in FPN models.
        num_category(int):
        num_convs (int): number of convolution layers
        norm (str or None): either None or 'GN'

    Returns:
        mask_logits (N x num_category x 2s x 2s):
    """
    assert norm in [None, 'GN'], norm
    l = feature
    with argscope([Conv2D, Conv2DTranspose], data_format='channels_first',
                  kernel_initializer=tfv1.variance_scaling_initializer(
                      scale=2.0, mode='fan_out',
                      distribution='untruncated_normal')):
        # c2's MSRAFill is fan_out
        for k in range(num_convs):
            l = Conv2D('fcn{}'.format(k), l, cfg.MRCNN.HEAD_DIM, 3, activation=tf.nn.relu)
            if norm is not None:
                l = GroupNorm('gn{}'.format(k), l)
        l = Conv2DTranspose('deconv', l, cfg.MRCNN.HEAD_DIM, 2, strides=2, activation=tf.nn.relu)
        l = Conv2D('conv', l, num_category, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    return l


def maskrcnn_up4conv_head(*args, **kwargs):
    return maskrcnn_upXconv_head(*args, num_convs=4, **kwargs)


def maskrcnn_up4conv_gn_head(*args, **kwargs):
    return maskrcnn_upXconv_head(*args, num_convs=4, norm='GN', **kwargs)


def unpackbits_masks(masks):
    """
    Args:
        masks (Tensor): uint8 Tensor of shape N, H, W. The last dimension is packed bits.

    Returns:
        masks (Tensor): bool Tensor of shape N, H, 8*W.

    This is a reverse operation of `np.packbits`
    """
    assert masks.dtype == tf.uint8, masks
    bits = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8)
    unpacked = tf.bitwise.bitwise_and(tf.expand_dims(masks, -1), bits) > 0
    unpacked = tf.reshape(
        unpacked,
        tf.concat([tf.shape(masks)[:-1], [8 * tf.shape(masks)[-1]]], axis=0))
    return unpacked
