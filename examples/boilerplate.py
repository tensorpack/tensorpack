# -*- coding: utf-8 -*-
# Author: Your Name <your@email.com>

import argparse
import os
import tensorflow as tf

from tensorpack import *

"""
This is a boiler-plate template.
All code is in this file is the most minimalistic way to solve a deep-learning problem with cross-validation.
"""

BATCH_SIZE = 16
SHAPE = 28
CHANNELS = 3


class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec((None, SHAPE, SHAPE, CHANNELS), tf.float32, 'input1'),
                tf.TensorSpec((None,), tf.int32, 'input2')]

    def build_graph(self, input1, input2):

        cost = tf.identity(input1 - input2, name='total_costs')
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=5e-3, trainable=False)
        return tf.train.AdamOptimizer(lr)


def get_data(subset):
    # something that yields [[SHAPE, SHAPE, CHANNELS], [1]]
    ds = FakeData([[SHAPE, SHAPE, CHANNELS], [1]], 1000, random=False,
                  dtype=['float32', 'uint8'], domain=[(0, 255), (0, 10)])
    ds = MultiProcessRunnerZMQ(ds, 2)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def get_config():
    logger.auto_set_dir()

    ds_train = get_data('train')
    ds_test = get_data('test')

    return TrainConfig(
        model=Model(),
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_test, [ScalarStats('total_costs')]),
        ],
        steps_per_epoch=len(ds_train),
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    config.session_init = SmartInit(args.load)

    launch_train_with_config(config, SimpleTrainer())
