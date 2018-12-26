#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: shufflenet.py

import argparse
import math
import numpy as np
import os
import cv2
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader, model_utils
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import ImageNetModel, eval_on_ILSVRC12, get_imagenet_dataflow


@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
              W_init=None, activation=tf.identity):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1]
    assert out_channel % in_channel == 0, (out_channel, in_channel)
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
    l = tf.reshape(l, [-1, in_channel // group, group] + in_shape[-2:])
    l = tf.transpose(l, [0, 2, 1, 3, 4])
    l = tf.reshape(l, [-1, in_channel] + in_shape[-2:])
    return l


@layer_register()
def shufflenet_unit(l, out_channel, group, stride):
    in_shape = l.get_shape().as_list()
    in_channel = in_shape[1]
    shortcut = l

    # "We do not apply group convolution on the first pointwise layer
    #  because the number of input channels is relatively small."
    first_split = group if in_channel > 24 else 1
    l = Conv2D('conv1', l, out_channel // 4, 1, split=first_split, activation=BNReLU)
    l = channel_shuffle(l, group)
    l = DepthConv('dconv', l, out_channel // 4, 3, stride=stride)
    l = BatchNorm('dconv_bn', l)

    l = Conv2D('conv2', l,
               out_channel if stride == 1 else out_channel - in_channel,
               1, split=group)
    l = BatchNorm('conv2_bn', l)
    if stride == 1:     # unit (b)
        output = tf.nn.relu(shortcut + l)
    else:   # unit (c)
        shortcut = AvgPooling('avgpool', shortcut, 3, 2, padding='SAME')
        output = tf.concat([shortcut, tf.nn.relu(l)], axis=1)
    return output


@layer_register()
def shufflenet_unit_v2(l, out_channel, stride):
    if stride == 1:
        shortcut, l = tf.split(l, 2, axis=1)
    else:
        shortcut, l = l, l
    shortcut_channel = int(shortcut.shape[1])

    l = Conv2D('conv1', l, out_channel // 2, 1, activation=BNReLU)
    l = DepthConv('dconv', l, out_channel // 2, 3, stride=stride)
    l = BatchNorm('dconv_bn', l)
    l = Conv2D('conv2', l, out_channel - shortcut_channel, 1, activation=BNReLU)

    if stride == 2:
        shortcut = DepthConv('shortcut_dconv', shortcut, shortcut_channel, 3, stride=2)
        shortcut = BatchNorm('shortcut_dconv_bn', shortcut)
        shortcut = Conv2D('shortcut_conv', shortcut, shortcut_channel, 1, activation=BNReLU)
    output = tf.concat([shortcut, l], axis=1)
    output = channel_shuffle(output, 2)
    return output


@layer_register(log_shape=True)
def shufflenet_stage(input, channel, num_blocks, group):
    l = input
    for i in range(num_blocks):
        name = 'block{}'.format(i)
        if args.v2:
            l = shufflenet_unit_v2(name, l, channel, 2 if i == 0 else 1)
        else:
            l = shufflenet_unit(name, l, channel, group, 2 if i == 0 else 1)
    return l


class Model(ImageNetModel):
    weight_decay = 4e-5

    def get_logits(self, image):

        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='channels_first'), \
                argscope(Conv2D, use_bias=False):

            group = args.group
            if not args.v2:
                # Copied from the paper
                channels = {
                    3: [240, 480, 960],
                    4: [272, 544, 1088],
                    8: [384, 768, 1536]
                }
                mul = group * 4  # #chan has to be a multiple of this number
                channels = [int(math.ceil(x * args.ratio / mul) * mul)
                            for x in channels[group]]
                # The first channel must be a multiple of group
                first_chan = int(math.ceil(24 * args.ratio / group) * group)
            else:
                # Copied from the paper
                channels = {
                    0.5: [48, 96, 192],
                    1.: [116, 232, 464]
                }[args.ratio]
                first_chan = 24

            logger.info("#Channels: " + str([first_chan] + channels))

            l = Conv2D('conv1', image, first_chan, 3, strides=2, activation=BNReLU)
            l = MaxPooling('pool1', l, 3, 2, padding='SAME')

            l = shufflenet_stage('stage2', l, channels[0], 4, group)
            l = shufflenet_stage('stage3', l, channels[1], 8, group)
            l = shufflenet_stage('stage4', l, channels[2], 4, group)

            if args.v2:
                l = Conv2D('conv5', l, 1024, 1, activation=BNReLU)

            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, 1000)
            return logits


def get_data(name, batch):
    isTrain = name == 'train'

    if isTrain:
        augmentors = [
            # use lighter augs if model is too small
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.49 if args.ratio < 1 else 0.08, 1.)),
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
        EstimatedTimeLeft()
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
    parser.add_argument('-r', '--ratio', type=float, default=0.5, choices=[1., 0.5])
    parser.add_argument('--group', type=int, default=8, choices=[3, 4, 8],
                        help="Number of groups for ShuffleNetV1")
    parser.add_argument('--v2', action='store_true', help='Use ShuffleNetV2')
    parser.add_argument('--batch', type=int, default=1024, help='total batch size')
    parser.add_argument('--load', help='path to load a model from')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.v2 and args.group != parser.get_default('group'):
        logger.error("group= is not used in ShuffleNetV2!")

    if args.batch != 1024:
        logger.warn("Total batch size != 1024, you need to change other hyperparameters to get the same results.")
    TOTAL_BATCH_SIZE = args.batch

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
        with TowerContext('', is_training=False):
            model.build_graph(*input.get_input_tensors())
        model_utils.describe_trainable_vars()

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
        logger.info("Note that TensorFlow counts flops in a different way from the paper.")
        logger.info("TensorFlow counts multiply+add as two flops, however the paper counts them "
                    "as 1 flop because it can be executed in one instruction.")
    else:
        if args.v2:
            name = "ShuffleNetV2-{}x".format(args.ratio)
        else:
            name = "ShuffleNetV1-{}x-g{}".format(args.ratio, args.group)
        logger.set_logger_dir(os.path.join('train_log', name))

        nr_tower = max(get_num_gpu(), 1)
        config = get_config(model, nr_tower)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower))
