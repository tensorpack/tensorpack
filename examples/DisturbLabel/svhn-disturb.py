#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: svhn-disturb.py

import argparse
import imp
import os

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils import logger

from disturb import DisturbLabel

svhn_example = imp.load_source('svhn_example',
                               os.path.join(os.path.dirname(__file__), '..',
                                            'basics', 'svhn-digit-convnet.py'))
Model = svhn_example.Model


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob', help='disturb prob', type=float, required=True)
    args = parser.parse_args()

    logger.auto_set_dir()
    data_train, data_test = get_data()
    config = TrainConfig(
        model=Model(),
        data=QueueInput(data_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(data_test,
                            ScalarStats(['cost', 'accuracy']))
        ],
        max_epoch=350,
    )
    launch_train_with_config(config, SimpleTrainer())
