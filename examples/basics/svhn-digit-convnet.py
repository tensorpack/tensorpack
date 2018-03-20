#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn-digit-convnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse
import os


from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import *
import tensorflow as tf

"""
A very small SVHN convnet model (only 0.8m parameters).
About 2.3% validation error after 70 epochs. 2.15% after 150 epochs.

Each epoch iterates over the whole training set (4721 iterations).
Speed is about 43 it/s on TitanX.
"""


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 40, 40, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = image / 128.0 - 1

        with argscope(Conv2D, activation=BNReLU, use_bias=False):
            logits = (LinearWrap(image)
                      .Conv2D('conv1', 24, 5, padding='VALID')
                      .MaxPooling('pool1', 2, padding='SAME')
                      .Conv2D('conv2', 32, 3, padding='VALID')
                      .Conv2D('conv3', 32, 3, padding='VALID')
                      .MaxPooling('pool2', 2, padding='SAME')
                      .Conv2D('conv4', 64, 3, padding='VALID')
                      .Dropout('drop', rate=0.5)
                      .FullyConnected('fc0', 512,
                                      bias_initializer=tf.constant_initializer(0.1),
                                      activation=tf.nn.relu)
                      .FullyConnected('linear', units=10)())
        tf.nn.softmax(logits, name='output')

        accuracy = tf.to_float(tf.nn.in_top_k(logits, label, 1))
        add_moving_summary(tf.reduce_mean(accuracy, name='accuracy'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wd_cost = regularize_cost('fc.*/W', l2_regularizer(0.00001))
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram', 'rms']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=4721 * 60,
            decay_rate=0.2, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data():
    d1 = dataset.SVHNDigit('train')
    d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1, d2])
    data_test = dataset.SVHNDigit('test', shuffle=False)

    augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5, 1.5)),
        imgaug.GaussianDeform(  # this is slow. only use it when you have lots of cpus
            [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
            (40, 40), 0.2, 3),
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
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
        session_init=SaverRestore(args.load) if args.load else None
    )
    launch_train_with_config(config, SimpleTrainer())
