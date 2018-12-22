#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: alexnet.py

import argparse
import numpy as np
import os
import cv2
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import ImageNetModel, get_imagenet_dataflow


def visualize_conv1_weights(filters):
    ctx = get_current_tower_context()
    if not ctx.is_main_training_tower:
        return
    with tf.name_scope('visualize_conv1'):
        filters = tf.reshape(filters, [11, 11, 3, 8, 12])
        filters = tf.transpose(filters, [3, 0, 4, 1, 2])    # 8,11,12,11,3
        filters = tf.reshape(filters, [1, 88, 132, 3])
    tf.summary.image('visualize_conv1', filters, max_outputs=1, collections=['AAA'])


class Model(ImageNetModel):
    weight_decay = 5e-4
    data_format = 'NHWC'  # LRN only supports NHWC

    def get_logits(self, image):
        gauss_init = tf.random_normal_initializer(stddev=0.01)
        with argscope(Conv2D,
                      kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
                argscope([Conv2D, FullyConnected], activation=tf.nn.relu), \
                argscope([Conv2D, MaxPooling], data_format='channels_last'):
            # necessary padding to get 55x55 after conv1
            image = tf.pad(image, [[0, 0], [2, 2], [2, 2], [0, 0]])
            l = Conv2D('conv1', image, filters=96, kernel_size=11, strides=4, padding='VALID')
            # size: 55
            visualize_conv1_weights(l.variables.W)
            l = tf.nn.lrn(l, 2, bias=1.0, alpha=2e-5, beta=0.75, name='norm1')
            l = MaxPooling('pool1', l, 3, strides=2, padding='VALID')
            # 27
            l = Conv2D('conv2', l, filters=256, kernel_size=5, split=2)
            l = tf.nn.lrn(l, 2, bias=1.0, alpha=2e-5, beta=0.75, name='norm2')
            l = MaxPooling('pool2', l, 3, strides=2, padding='VALID')
            # 13
            l = Conv2D('conv3', l, filters=384, kernel_size=3)
            l = Conv2D('conv4', l, filters=384, kernel_size=3, split=2)
            l = Conv2D('conv5', l, filters=256, kernel_size=3, split=2)
            l = MaxPooling('pool3', l, 3, strides=2, padding='VALID')

            l = FullyConnected('fc6', l, 4096,
                               kernel_initializer=gauss_init,
                               bias_initializer=tf.ones_initializer())
            l = Dropout(l, rate=0.5)
            l = FullyConnected('fc7', l, 4096, kernel_initializer=gauss_init)
            l = Dropout(l, rate=0.5)
        logits = FullyConnected('fc8', l, 1000, kernel_initializer=gauss_init)
        return logits


def get_data(name, batch):
    isTrain = name == 'train'
    if isTrain:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.RandomCrop(224),
            imgaug.Lighting(0.1,
                            eigval=np.asarray(
                                [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                            eigvec=np.array(
                                [[-0.5675, 0.7192, 0.4009],
                                 [-0.5808, -0.0045, -0.8140],
                                 [-0.5836, -0.6948, 0.4203]],
                                dtype='float32')[::-1, ::-1]),
            imgaug.Flip(horiz=True)]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224))]
    return get_imagenet_dataflow(args.data, name, batch, augmentors)


def get_config():
    nr_tower = max(get_num_gpu(), 1)
    batch = args.batch
    total_batch = batch * nr_tower
    if total_batch != 128:
        logger.warn("AlexNet needs to be trained with a total batch size of 128.")
    BASE_LR = 0.01 * (total_batch / 128.)

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
            [(0, BASE_LR), (30, BASE_LR * 1e-1), (60, BASE_LR * 1e-2), (80, BASE_LR * 1e-3)]),
        DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))),
    ]

    return TrainConfig(
        model=Model(),
        data=StagingInput(QueueInput(dataset_train)),
        callbacks=callbacks,
        steps_per_epoch=1281167 // total_batch,
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--batch', type=int, default=32, help='batch per GPU')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_logger_dir(os.path.join('train_log', 'AlexNet'))

    config = get_config()
    nr_tower = max(get_num_gpu(), 1)
    trainer = SyncMultiGPUTrainerReplicated(nr_tower)
    launch_train_with_config(config, trainer)
