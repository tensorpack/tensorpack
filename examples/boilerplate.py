#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Your Name <your@email.com>

import os
import argparse
from tensorpack import *
import tensorflow as tf

"""
This is a boiler-plate template.
All code is in this file is the most minimalistic way to solve a deep-learning problem with cross-validation.
"""

BATCH_SIZE = 16
SHAPE = 28
CHANNELS = 3


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, SHAPE, SHAPE, CHANNELS), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image * 2 - 1

        self.cost = tf.identity(0., name='total_costs')
        summary.add_moving_summary(self.cost)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 5e-3, summary=True)
        return tf.train.AdamOptimizer(lr)


def get_data(subset):
    # something that yields [[SHAPE, SHAPE, CHANNELS], [1]]
    ds = FakeData([[SHAPE, SHAPE, CHANNELS], [1]], 1000, random=False,
                  dtype=['float32', 'uint8'], domain=[(0, 255), (0, 10)])
    ds = PrefetchDataZMQ(ds, 2)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def get_config():
    logger.auto_set_dir()

    ds_train = get_data('train')
    ds_test = get_data('test')

    return TrainConfig(
        model=Model(),
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_test, [ScalarStats('total_costs')]),
        ],
        steps_per_epoch=ds_train.size(),
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', required=True)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    NR_GPU = len(args.gpu.split(','))
    with change_gpu(args.gpu):
        config = get_config()
        config.nr_tower = NR_GPU

        if args.load:
            config.session_init = SaverRestore(args.load)

        SyncMultiGPUTrainer(config).train()
