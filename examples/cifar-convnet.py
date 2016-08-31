#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar-convnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack import *
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.summary import *

"""
A small convnet model for Cifar10 or Cifar100 dataset.

Cifar10:
    90% validation accuracy after 40k step.
    91% accuracy after 80k step.
    19.3 step/s on Tesla M40

Not a good model for Cifar100, just for demonstration.
"""

class Model(ModelDesc):
    def __init__(self, cifar_classnum):
        super(Model, self).__init__()
        self.cifar_classnum = cifar_classnum

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 30, 30, 3], 'input'),
                InputVar(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        is_training = get_current_tower_context().is_training
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        if is_training:
            tf.image_summary("train_image", image, 10)

        image = image / 4.0     # just to make range smaller
        with argscope(Conv2D, nl=BNReLU(), use_bias=False, kernel_shape=3):
            logits = LinearWrap(image) \
                    .Conv2D('conv1.1', out_channel=64) \
                    .Conv2D('conv1.2', out_channel=64) \
                    .MaxPooling('pool1', 3, stride=2, padding='SAME') \
                    .Conv2D('conv2.1', out_channel=128) \
                    .Conv2D('conv2.2', out_channel=128) \
                    .MaxPooling('pool2', 3, stride=2, padding='SAME') \
                    .Conv2D('conv3.1', out_channel=128, padding='VALID') \
                    .Conv2D('conv3.2', out_channel=128, padding='VALID') \
                    .FullyConnected('fc0', 1024 + 512,
                           b_init=tf.constant_initializer(0.1)) \
                    .tf.nn.dropout(keep_prob) \
                    .FullyConnected('fc1', 512,
                           b_init=tf.constant_initializer(0.1)) \
                    .FullyConnected('linear', out_dim=self.cifar_classnum, nl=tf.identity)()

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = symbf.prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.004,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

def get_data(train_or_test, cifar_classnum):
    isTrain = train_or_test == 'train'
    if cifar_classnum == 10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.RandomCrop((30, 30)),
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2,1.8)),
            imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
                (30,30), 0.2, 3),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((30, 30)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config(cifar_classnum):
    logger.auto_set_dir()

    # prepare dataset
    dataset_train = get_data('train', cifar_classnum)
    step_per_epoch = dataset_train.size()
    dataset_test = get_data('test', cifar_classnum)

    sess_config = get_default_sess_config(0.5)

    nr_gpu = get_nr_gpu()
    lr = tf.train.exponential_decay(
        learning_rate=1e-2,
        global_step=get_global_step_var(),
        decay_steps=step_per_epoch * (30 if nr_gpu == 1 else 20),
        decay_rate=0.5, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(dataset_test, ClassificationError())
        ]),
        session_config=sess_config,
        model=Model(cifar_classnum),
        step_per_epoch=step_per_epoch,
        max_epoch=250,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--classnum', help='10 for cifar10 or 100 for cifar100',
                        type=int, default=10)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Graph().as_default():
        config = get_config(args.classnum)
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        QueueInputTrainer(config).train()
        #AsyncMultiGPUTrainer(config).train()
