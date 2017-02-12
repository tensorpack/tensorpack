#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn-resnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
ResNet-110 for SVHN Digit Classification.
Reach 1.8% validation error after 70 epochs, with 2 TitanX. 2it/s.
You might need to adjust the learning rate schedule when running with 1 GPU.
"""

import imp
cifar_example = imp.load_source('cifar_example',
                                os.path.join(os.path.dirname(__file__), 'cifar10-resnet.py'))
Model = cifar_example.Model

BATCH_SIZE = 128


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    pp_mean = dataset.SVHNDigit.get_per_pixel_mean()
    if isTrain:
        d1 = dataset.SVHNDigit('train')
        d2 = dataset.SVHNDigit('extra')
        ds = RandomMixData([d1, d2])
    else:
        ds = dataset.SVHNDigit('test')

    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.Brightness(10),
            imgaug.Contrast((0.8, 1.2)),
            imgaug.GaussianDeform(  # this is slow. without it, can only reach 1.9% error
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (40, 40), 0.2, 3),
            imgaug.RandomCrop((32, 32)),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 5, 5)
    return ds


def get_config():
    logger.auto_set_dir()

    # prepare dataset
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (20, 0.01), (28, 0.001), (50, 0.0001)])
        ],
        model=Model(n=18),
        steps_per_epoch=steps_per_epoch,
        max_epoch=500,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
