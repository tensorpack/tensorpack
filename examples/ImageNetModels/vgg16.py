#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: vgg16.py

import argparse
import os

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import (
    ImageNetModel, get_imagenet_dataflow, fbresnet_augmentor)


def convnormrelu(x, name, chan):
    x = Conv2D(name, x, chan, 3)
    if args.norm == 'bn':
        x = BatchNorm(name + '_bn', x)
    x = tf.nn.relu(x, name=name + '_relu')
    return x


class Model(ImageNetModel):
    weight_decay = 5e-4

    def get_logits(self, image):
        with argscope(Conv2D, kernel_size=3,
                      kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
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
    global args
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(args.data, name, batch, augmentors)


def get_config():
    nr_tower = max(get_nr_gpu(), 1)
    batch = 64
    total_batch = batch * nr_tower
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
    parser.add_argument('--norm', choices=['none', 'bn'], default='none')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_logger_dir(os.path.join('train_log', 'vgg16-norm={}'.format(args.norm)))

    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    nr_tower = max(get_nr_gpu(), 1)
    trainer = SyncMultiGPUTrainerReplicated(nr_tower)
    launch_train_with_config(config, trainer)
