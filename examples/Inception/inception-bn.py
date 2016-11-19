#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: inception-bn.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cv2
import argparse
import numpy as np
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *


TOTAL_BATCH_SIZE = 64 * 6
NR_GPU = 6
BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU
INPUT_SHAPE = 224

"""
Inception-BN model on ILSVRC12.
See "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", arxiv:1502.03167

This config reaches 71% single-crop validation accuracy after 150k steps with 6 TitanX.
Learning rate may need a different schedule for different number of GPUs (because batch size will be different).
"""

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 128.0

        def inception(name, x, nr1x1, nr3x3r, nr3x3, nr233r, nr233, nrpool, pooltype):
            stride = 2 if nr1x1 == 0 else 1
            with tf.variable_scope(name) as scope:
                outs = []
                if nr1x1 != 0:
                    outs.append(Conv2D('conv1x1', x, nr1x1, 1))
                x2 = Conv2D('conv3x3r', x, nr3x3r, 1)
                outs.append(Conv2D('conv3x3', x2, nr3x3, 3, stride=stride))

                x3 = Conv2D('conv233r', x, nr233r, 1)
                x3 = Conv2D('conv233a', x3, nr233, 3)
                outs.append(Conv2D('conv233b', x3, nr233, 3, stride=stride))

                if pooltype == 'max':
                    x4 = MaxPooling('mpool', x, 3, stride, padding='SAME')
                else:
                    assert pooltype == 'avg'
                    x4 = AvgPooling('apool', x, 3, stride, padding='SAME')
                if nrpool != 0:  # pool + passthrough if nrpool == 0
                    x4 = Conv2D('poolproj', x4, nrpool, 1)
                outs.append(x4)
                return tf.concat(3, outs, name='concat')

        with argscope(Conv2D, nl=BNReLU, use_bias=False):
            l = Conv2D('conv0', image, 64, 7, stride=2)
            l = MaxPooling('pool0', l, 3, 2, padding='SAME')
            l = Conv2D('conv1', l, 64, 1)
            l = Conv2D('conv2', l, 192, 3)
            l = MaxPooling('pool2', l, 3, 2, padding='SAME')
            # 28
            l = inception('incep3a', l, 64, 64, 64, 64, 96, 32, 'avg')
            l = inception('incep3b', l, 64, 64, 96, 64, 96, 64, 'avg')
            l = inception('incep3c', l, 0, 128, 160, 64, 96, 0, 'max')

            br1 = Conv2D('loss1conv', l, 128, 1)
            br1 = FullyConnected('loss1fc', br1, 1024, nl=tf.nn.relu)
            br1 = FullyConnected('loss1logit', br1, 1000, nl=tf.identity)
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(br1, label)
            loss1 = tf.reduce_mean(loss1, name='loss1')

            # 14
            l = inception('incep4a', l, 224, 64, 96, 96, 128, 128, 'avg')
            l = inception('incep4b', l, 192, 96, 128, 96, 128, 128, 'avg')
            l = inception('incep4c', l, 160, 128, 160, 128, 160, 128, 'avg')
            l = inception('incep4d', l, 96, 128, 192, 160, 192, 128, 'avg')
            l = inception('incep4e', l, 0, 128, 192, 192, 256, 0, 'max')

            br2 = Conv2D('loss2conv', l, 128, 1)
            br2 = FullyConnected('loss2fc', br2, 1024, nl=tf.nn.relu)
            br2 = FullyConnected('loss2logit', br2, 1000, nl=tf.identity)
            loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(br2, label)
            loss2 = tf.reduce_mean(loss2, name='loss2')

            # 7
            l = inception('incep5a', l, 352, 192, 320, 160, 224, 128, 'avg')
            l = inception('incep5b', l, 352, 192, 320, 192, 224, 128, 'max')
            l = GlobalAvgPooling('gap', l)

        logits = FullyConnected('linear', l, out_dim=1000, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')
        loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        loss3 = tf.reduce_mean(loss3, name='loss3')

        cost = tf.add_n([loss3, 0.3 * loss2, 0.3 * loss1], name='weighted_cost')
        add_moving_summary([cost, loss1, loss2, loss3])

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train_error_top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train_error_top5'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          80000, 0.7, True)
        wd_cost = tf.mul(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='l2_regularize_loss')
        add_moving_summary(wd_cost)

        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.ILSVRC12(args.data, train_or_test, shuffle=True if isTrain else False)
    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()

    if isTrain:
        # TODO use the augmentor in GoogleNet
        augmentors = [
            imgaug.Resize((256, 256)),
            imgaug.Brightness(30, False),
            imgaug.Contrast((0.8,1.2), True),
            imgaug.MapImage(lambda x: x - pp_mean),
            imgaug.RandomCrop((224, 224)),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.Resize((256, 256)),
            imgaug.MapImage(lambda x: x - pp_mean),
            imgaug.CenterCrop((224, 224)),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 6)
    return ds


def get_config():
    logger.auto_set_dir()
    # prepare dataset
    dataset_train = get_data('train')
    step_per_epoch = 5000
    dataset_val = get_data('val')

    lr = get_scalar_var('learning_rate', 0.045, summary=True)
    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            InferenceRunner(dataset_val, [
                ClassificationError('wrong-top1', 'val-top1-error'),
                ClassificationError('wrong-top5', 'val-top5-error')]),
            #HumanHyperParamSetter('learning_rate', 'hyper-googlenet.txt')
            ScheduledHyperParamSetter('learning_rate',
                                      [(8, 0.03), (14, 0.02), (17, 5e-3),
                                       (19, 3e-3), (24, 1e-3), (26, 2e-4),
                                       (30, 5e-5) ])
        ]),
        session_config=get_default_sess_config(0.99),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=80,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--data', help='ImageNet data root directory', required=True)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
