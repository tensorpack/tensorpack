#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_resnet_utils.py

import numpy as np
import cv2
import multiprocessing
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer


import tensorpack as tp
from tensorpack import imgaug, dataset
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, ThreadedMapData)
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU,
    LinearWrap)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor


class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(0.08, 1.0) * area
            aspectR = self.rng.uniform(0.75, 1.333)
            ww = int(np.sqrt(targetArea * aspectR))
            hh = int(np.sqrt(targetArea / aspectR))
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(224, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(224).augment(out)
        return out


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            GoogleNetResize(),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_imagenet_dataflow(
        datadir, name, batch_size,
        augmentors, dir_structure='original'):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    isTrain = name == 'train'
    cpu = min(30, multiprocessing.cpu_count())
    if isTrain:
        ds = dataset.ILSVRC12(datadir, name, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        ds = PrefetchDataZMQ(ds, cpu)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, name,
                                   shuffle=False, dir_structure=dir_structure)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls
        ds = ThreadedMapData(ds, cpu, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def resnet_shortcut(l, n_out, stride):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, stride=stride)
    else:
        return l


def apply_preactivation(l, preact):
    """
    'no_preact' for the first resblock only, because the input is activated already.
    'both_preact' for the first block in each group, due to the projection shotcut.
    'default' for all the non-first blocks, where identity mapping is preserved on shortcut path.
    """
    if preact == 'both_preact':
        l = BNReLU('preact', l)
        shortcut = l
    elif preact == 'default':
        shortcut = l
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def resnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def resnet_bottleneck(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def resnet_group(l, name, block_func, features, count, stride, first=False):
    with tf.variable_scope(name):
        with tf.variable_scope('block0'):
            l = block_func(l, features, stride,
                           'no_preact' if first else 'both_preact')
        for i in range(1, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, 1, 'default')
        return l


def resnet_backbone(image, num_blocks, block_func):
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=variance_scaling_initializer(mode='FAN_OUT')):
        logits = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(resnet_group, 'group0', block_func, 64, num_blocks[0], 1, first=True)
                  .apply(resnet_group, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(resnet_group, 'group2', block_func, 256, num_blocks[2], 2)
                  .apply(resnet_group, 'group3', block_func, 512, num_blocks[3], 2)
                  .BNReLU('bnlast')
                  .GlobalAvgPooling('gap')
                  .FullyConnected('linear', 1000, nl=tf.identity)())
    return logits


def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        acc5.feed(o[1].sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


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


def compute_loss_and_error(logits, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    loss = tf.reduce_mean(loss, name='xentropy-loss')

    def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
        with tf.name_scope('prediction_incorrect'):
            x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
        return tf.cast(x, tf.float32, name=name)

    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

    wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
    return loss
