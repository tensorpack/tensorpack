#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn-disturb.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from disturb import DisturbLabel
import imp

svhn_example = imp.load_source('svhn_example',
                               os.path.join(os.path.dirname(__file__), '..', 'svhn-digit-convnet.py'))
Model = svhn_example.Model
get_config = svhn_example.get_config


def get_data():
    d1 = dataset.SVHNDigit('train')
    d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1, d2])
    data_train = DisturbLabel(data_train, args.prob)
    data_test = dataset.SVHNDigit('test')

    augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5, 1.5)),
    ]
    data_train = AugmentImageComponent(data_train, augmentors)
    data_train = BatchData(data_train, 128)
    data_train = PrefetchData(data_train, 5, 5)

    augmentors = [imgaug.Resize((40, 40))]
    data_test = AugmentImageComponent(data_test, augmentors)
    data_test = BatchData(data_test, 128, remainder=True)
    return data_train, data_test


svhn_example.get_data = get_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='a gpu to use')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--prob', help='disturb prob',
                        type=float, required=True)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = get_config(args.prob)
    if args.load:
        config.session_init = SaverRestore(args.load)
    QueueInputTrainer(config).train()
