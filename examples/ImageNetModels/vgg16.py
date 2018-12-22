#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg16.py

import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils import argscope
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import ImageNetModel, fbresnet_augmentor, get_imagenet_dataflow


def GroupNorm(x, group, gamma_initializer=tf.constant_initializer(1.)):
    """
    https://arxiv.org/abs/1803.08494
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


def convnormrelu(x, name, chan):
    x = Conv2D(name, x, chan, 3)
    if args.norm == 'bn':
        x = BatchNorm(name + '_bn', x)
    elif args.norm == 'gn':
        with tf.variable_scope(name + '_gn'):
            x = GroupNorm(x, 32)
    x = tf.nn.relu(x, name=name + '_relu')
    return x


class Model(ImageNetModel):
    weight_decay = 5e-4

    def get_logits(self, image):
        with argscope(Conv2D, kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
                argscope([Conv2D, MaxPooling, BatchNorm], data_format='channels_first'):
            logits = (LinearWrap(image)
                      .apply(convnormrelu, 'conv1_1', 64)
                      .apply(convnormrelu, 'conv1_2', 64)
                      .MaxPooling('pool1', 2)
                      # 112
                      .apply(convnormrelu, 'conv2_1', 128)
                      .apply(convnormrelu, 'conv2_2', 128)
                      .MaxPooling('pool2', 2)
                      # 56
                      .apply(convnormrelu, 'conv3_1', 256)
                      .apply(convnormrelu, 'conv3_2', 256)
                      .apply(convnormrelu, 'conv3_3', 256)
                      .MaxPooling('pool3', 2)
                      # 28
                      .apply(convnormrelu, 'conv4_1', 512)
                      .apply(convnormrelu, 'conv4_2', 512)
                      .apply(convnormrelu, 'conv4_3', 512)
                      .MaxPooling('pool4', 2)
                      # 14
                      .apply(convnormrelu, 'conv5_1', 512)
                      .apply(convnormrelu, 'conv5_2', 512)
                      .apply(convnormrelu, 'conv5_3', 512)
                      .MaxPooling('pool5', 2)
                      # 7
                      .FullyConnected('fc6', 4096,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.001))
                      .tf.nn.relu(name='fc6_relu')
                      .Dropout('drop0', rate=0.5)
                      .FullyConnected('fc7', 4096,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.001))
                      .tf.nn.relu(name='fc7_relu')
                      .Dropout('drop1', rate=0.5)
                      .FullyConnected('fc8', 1000,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01))())
        add_param_summary(('.*', ['histogram', 'rms']))
        return logits


def get_data(name, batch):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(args.data, name, batch, augmentors)


def get_config():
    nr_tower = max(get_num_gpu(), 1)
    batch = args.batch
    total_batch = batch * nr_tower
    assert total_batch >= 256   # otherwise the learning rate warmup is wrong.
    BASE_LR = 0.01 * (total_batch / 256.)

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch)
    dataset_val = get_data('val', batch)

    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    callbacks = [
        ModelSaver(),
        GPUUtilizationTracker(),
        EstimatedTimeLeft(),
        ScheduledHyperParamSetter(
            'learning_rate',
            [(0, 0.01), (3, max(BASE_LR, 0.01))], interp='linear'),
        ScheduledHyperParamSetter(
            'learning_rate',
            [(30, BASE_LR * 1e-1), (60, BASE_LR * 1e-2), (80, BASE_LR * 1e-3)]),
        DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))),
    ]

    input = QueueInput(dataset_train)
    input = StagingInput(input, nr_stage=1)
    return TrainConfig(
        model=Model(),
        data=input,
        callbacks=callbacks,
        steps_per_epoch=1281167 // total_batch,
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--batch', type=int, default=32, help='batch per GPU')
    parser.add_argument('--norm', choices=['none', 'bn', 'gn'], default='none')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_logger_dir(os.path.join('train_log', 'vgg16-norm={}'.format(args.norm)))

    config = get_config()
    nr_tower = max(get_num_gpu(), 1)
    trainer = SyncMultiGPUTrainerReplicated(nr_tower)
    launch_train_with_config(config, trainer)
