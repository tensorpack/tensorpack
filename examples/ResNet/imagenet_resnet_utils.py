#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_resnet_utils.py

import numpy as np
import cv2
import multiprocessing
from abc import abstractmethod
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer


from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, ThreadedMapData)
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap, regularize_cost)
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
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
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
        augmentors):
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
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
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


def resnet_shortcut(l, n_out, stride, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, stride=stride, nl=nl)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        # this is used only for preact-resnet
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)


def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out, stride, nl=get_bn(zero_init=False))


def resnet_bottleneck(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


def se_resnet_bottleneck(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
    l = l * tf.reshape(squeeze, [-1, ch_out * 4, 1, 1])
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features,
                               stride if i == 0 else 1, 'no_preact')
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=variance_scaling_initializer(mode='FAN_OUT')):
        logits = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                  .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                  .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2)
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
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
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


class ImageNetModel(ModelDesc):
    def __init__(self, data_format='NCHW', image_dtype=tf.uint8):
        if data_format == 'NCHW':
            assert tf.test.is_gpu_available()
        self.data_format = data_format

        # uint8 instead of float32 is used as input type to reduce copy overhead.
        # It might hurt the performance a liiiitle bit.
        # The pretrained models were trained with float32.
        self.image_dtype = image_dtype

    def _get_inputs(self):
        return [InputDesc(self.image_dtype, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image_preprocess(image, bgr=True)
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)
        loss = compute_loss_and_error(logits, label)
        wd_loss = regularize_cost('.*/W', tf.contrib.layers.l2_regularizer(1e-4), name='l2_regularize_loss')
        add_moving_summary(loss, wd_loss)
        self.cost = tf.add_n([loss, wd_loss], name='cost')

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of 224x224 in ``self.data_format``

        Returns:
            Bx1000 logits
        """

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
