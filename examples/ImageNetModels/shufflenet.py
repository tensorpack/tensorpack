#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: shufflenet.py

import argparse
import numpy as np
import os
import cv2

import tensorflow as tf


from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader, model_utils
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import (
    get_imagenet_dataflow,
    ImageNetModel, GoogleNetResize, eval_on_ILSVRC12)

TOTAL_BATCH_SIZE = 1024


@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
              W_init=None, activation=tf.identity):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1]
    assert out_channel % in_channel == 0
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = tf.variance_scaling_initializer(2.0)
    kernel_shape = [kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, 1, stride, stride], padding=padding, data_format='NCHW')
    return activation(conv, name='output')


@under_name_scope()
def channel_shuffle(l, group):
    in_shape = l.get_shape().as_list()
    in_channel = in_shape[1]
    assert in_channel % group == 0, in_channel
    l = tf.reshape(l, [-1, group, in_channel // group] + in_shape[-2:])
    l = tf.transpose(l, [0, 2, 1, 3, 4])
    l = tf.reshape(l, [-1, in_channel] + in_shape[-2:])
    return l


def BN(x, name=None):
    return BatchNorm('bn', x)


class Model(ImageNetModel):
    weight_decay = 4e-5

    def get_logits(self, image):
        def shufflenet_unit(l, out_channel, group, stride):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[1]
            shortcut = l

            # We do not apply group convolution on the first pointwise layer
            # because the number of input channels is relatively small.
            first_split = group if in_channel != 12 else 1
            l = Conv2D('conv1', l, out_channel // 4, 1, split=first_split, activation=BNReLU)
            l = channel_shuffle(l, group)
            l = DepthConv('dconv', l, out_channel // 4, 3, activation=BN, stride=stride)

            l = Conv2D('conv2', l,
                       out_channel if stride == 1 else out_channel - in_channel,
                       1, split=group, activation=BN)
            if stride == 1:     # unit (b)
                output = tf.nn.relu(shortcut + l)
            else:   # unit (c)
                shortcut = AvgPooling('avgpool', shortcut, 3, 2, padding='SAME')
                output = tf.concat([shortcut, tf.nn.relu(l)], axis=1)
            return output

        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format), \
                argscope(Conv2D, use_bias=False):
            group = 3
            channels = [120, 240, 480]

            l = Conv2D('conv1', image, 12, 3, strides=2, activation=BNReLU)
            l = MaxPooling('pool1', l, 3, 2, padding='SAME')

            with tf.variable_scope('group1'):
                for i in range(4):
                    with tf.variable_scope('block{}'.format(i)):
                        l = shufflenet_unit(l, channels[0], group, 2 if i == 0 else 1)

            with tf.variable_scope('group2'):
                for i in range(8):
                    with tf.variable_scope('block{}'.format(i)):
                        l = shufflenet_unit(l, channels[1], group, 2 if i == 0 else 1)

            with tf.variable_scope('group3'):
                for i in range(4):
                    with tf.variable_scope('block{}'.format(i)):
                        l = shufflenet_unit(l, channels[2], group, 2 if i == 0 else 1)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, 1000)
            return logits


def get_data(name, batch):
    isTrain = name == 'train'

    if isTrain:
        augmentors = [
            GoogleNetResize(crop_area_fraction=0.49),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
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
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors)


def get_config(model, nr_tower):
    batch = TOTAL_BATCH_SIZE // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch)
    dataset_val = get_data('val', batch)

    step_size = 1280000 // TOTAL_BATCH_SIZE
    max_iter = 3 * 10**5
    max_epoch = (max_iter // step_size) + 1
    callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, 0.5), (max_iter, 0)],
                                  interp='linear', step_based=True),
    ]
    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    if nr_tower == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=step_size,
        max_epoch=max_epoch,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model()

    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    elif args.flops:
        # manually build the graph with batch=1
        input_desc = [
            InputDesc(tf.float32, [1, 224, 224, 3], 'input'),
            InputDesc(tf.int32, [1], 'label')
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=True):
            model.build_graph(*input.get_input_tensors())
        model_utils.describe_trainable_vars()

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
    else:
        logger.set_logger_dir(os.path.join('train_log', 'shufflenet'))

        nr_tower = max(get_nr_gpu(), 1)
        config = get_config(model, nr_tower)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower))
