#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10-convnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import numpy
import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack.train import TrainConfig, QueueInputTrainer
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.utils import *
from tensorpack.tfutils import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import *

"""
A small convnet model for cifar 10 or cifar100 dataset.
90% validation accuracy after 40k step.
"""

class Model(ModelDesc):
    def __init__(self, cifar_classnum):
        super(Model, self).__init__()
        self.cifar_classnum = cifar_classnum

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 30, 30, 3], 'input'),
                InputVar(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars, is_training):
        image, label = input_vars
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        if is_training:
            tf.image_summary("train_image", image, 10)

        image = image / 4.0     # just to make range smaller
        with argscope(Conv2D, nl=BNReLU(is_training), use_bias=False, kernel_shape=3):
            l = Conv2D('conv1.1', image, out_channel=64)
            l = Conv2D('conv1.2', l, out_channel=64)
            l = MaxPooling('pool1', l, 3, stride=2, padding='SAME')

            l = Conv2D('conv2.1', l, out_channel=128)
            l = Conv2D('conv2.2', l, out_channel=128)
            l = MaxPooling('pool2', l, 3, stride=2, padding='SAME')

            l = Conv2D('conv3.1', l, out_channel=128, padding='VALID')
            l = Conv2D('conv3.2', l, out_channel=128, padding='VALID')
        l = FullyConnected('fc0', l, 1024 + 512,
                           b_init=tf.constant_initializer(0.1))
        l = tf.nn.dropout(l, keep_prob)
        l = FullyConnected('fc1', l, 512,
                           b_init=tf.constant_initializer(0.1))
        # fc will have activation summary by default. disable for the output layer
        logits = FullyConnected('linear', l, out_dim=self.cifar_classnum, nl=tf.identity)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost)

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        tf.add_to_collection(
            MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.004,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)

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
        ds = PrefetchData(ds, 10, 5)
    return ds

def get_config(cifar_classnum):
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
    parser.add_argument('--classnum', help='specify cifar10 or cifar100, input 10 for cifar10 or 100 for cifar100')
    args = parser.parse_args()

    basename = os.path.basename(__file__)
    logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.classnum:
        cifar_classnum = int(args.classnum)
    else:
        cifar_classnum = 10

    with tf.Graph().as_default():
        config = get_config(cifar_classnum)
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        QueueInputTrainer(config).train()
