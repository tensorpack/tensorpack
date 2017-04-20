#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import os
import sys
import argparse

"""
python mnist-distributed.py --task_id 0 --job ps --gpu 0
python mnist-distributed.py --task_id 0 --job worker --gpu 0
python mnist-distributed.py --task_id 1 --job worker --gpu 0



"""

# Just import everything into current namespace
from tensorpack import *
import tensorflow as tf
import tensorflow.contrib.slim as slim

IMAGE_SIZE = 28


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):

        image, label = inputs
        image = tf.expand_dims(image, 3)
        image = image * 2 - 1   # center the pixels values at zero

        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
            logits = (LinearWrap(image)  # the starting brace is only for line-breaking
                      .Conv2D('conv0')
                      .MaxPooling('pool0', 2)
                      .Conv2D('conv1')
                      .Conv2D('conv2')
                      .MaxPooling('pool1', 2)
                      .Conv2D('conv3')
                      .FullyConnected('fc0', 512, nl=tf.nn.relu)
                      .Dropout('dropout', 0.5)
                      .FullyConnected('fc1', out_dim=10, nl=tf.identity)())

        prob = tf.nn.softmax(logits, name='prob')   # a Bx10 with probabilities

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        wrong = symbolic_functions.prediction_incorrect(logits, label, name='incorrect')

        train_error = tf.reduce_mean(wrong, name='train_error')
        # summary.add_moving_summary(train_error)

        self.cost = cost

    def _get_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)

        num_workers = 2
        opt = tf.train.AdamOptimizer(lr)
        opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_workers,
                                             total_num_replicas=num_workers)
        return opt


def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return train, test


def get_config():
    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data()
    steps_per_epoch = dataset_train.size()

    MACHINE_Q = 'xxx.x.xx.xxx'  # noqa
    MACHINE_G = 'xxx.x.xx.xxx'  # noqa

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,  # the DataFlow instance for training
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            # InferenceRunner(    # run inference(for validation) after every epoch
            #     dataset_test,   # the DataFlow instance used for validation
            #     # Calculate both the cost and the error for this DataFlow
            #     [ScalarStats('cross_entropy_loss'), ClassificationError('incorrect')]),
        ],
        cluster_spec={'ps': ['%s:2222' % MACHINE_Q],
                      'worker': ['%s:2223' % MACHINE_Q, '%s:2224' % MACHINE_G]},
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task_id', help='task identifier', required=True, type=int)
    parser.add_argument('--job', help='task identifier', required=True, type=str)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    DistributedTrainer(config, args.task_id, args.job).train()
