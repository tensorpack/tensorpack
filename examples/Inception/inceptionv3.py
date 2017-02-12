#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: inceptionv3.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cv2
import argparse
import numpy as np
import os
import tensorflow as tf
import multiprocessing

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
InceptionV3 on ILSVRC12.
See "Rethinking the Inception Architecture for Computer Vision", arxiv:1512.00567

This config follows the official inceptionv3 setup
(https://github.com/tensorflow/models/tree/master/inception/inception)
with much much fewer lines of code.
It reaches 74% single-crop validation accuracy, similar to the official code.

The hyperparameters here are for 8 GPUs, so the effective batch size is 8*64 = 512.
"""

TOTAL_BATCH_SIZE = 512
NR_GPU = 8
BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU
INPUT_SHAPE = 299


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 255.0   # ?

        def proj_kk(l, k, ch_r, ch, stride=1):
            l = Conv2D('conv{0}{0}r'.format(k), l, ch_r, 1)
            return Conv2D('conv{0}{0}'.format(k), l, ch, k, stride=stride,
                          padding='VALID' if stride > 1 else 'SAME')

        def proj_233(l, ch_r, ch, stride=1):
            l = Conv2D('conv233r', l, ch_r, 1)
            l = Conv2D('conv233a', l, ch, 3)
            return Conv2D('conv233b', l, ch, 3, stride=stride,
                          padding='VALID' if stride > 1 else 'SAME')

        def pool_proj(l, ch, pool_type):
            if pool_type == 'max':
                l = MaxPooling('maxpool', l, 3, 1)
            else:
                l = AvgPooling('maxpool', l, 3, 1, padding='SAME')
            return Conv2D('poolproj', l, ch, 1)

        def proj_77(l, ch_r, ch):
            return (LinearWrap(l)
                    .Conv2D('conv77r', ch_r, 1)
                    .Conv2D('conv77a', ch_r, [1, 7])
                    .Conv2D('conv77b', ch, [7, 1])())

        def proj_277(l, ch_r, ch):
            return (LinearWrap(l)
                    .Conv2D('conv277r', ch_r, 1)
                    .Conv2D('conv277aa', ch_r, [7, 1])
                    .Conv2D('conv277ab', ch_r, [1, 7])
                    .Conv2D('conv277ba', ch_r, [7, 1])
                    .Conv2D('conv277bb', ch, [1, 7])())

        with argscope(Conv2D, nl=BNReLU, use_bias=False),\
                argscope(BatchNorm, decay=0.9997, epsilon=1e-3):
            l = (LinearWrap(image)
                 .Conv2D('conv0', 32, 3, stride=2, padding='VALID')  # 299
                 .Conv2D('conv1', 32, 3, padding='VALID')  # 149
                 .Conv2D('conv2', 64, 3, padding='SAME')  # 147
                 .MaxPooling('pool2', 3, 2)
                 .Conv2D('conv3', 80, 1, padding='SAME')  # 73
                 .Conv2D('conv4', 192, 3, padding='VALID')  # 71
                 .MaxPooling('pool4', 3, 2)())  # 35

            with tf.variable_scope('incep-35-256a'):
                l = tf.concat([
                    Conv2D('conv11', l, 64, 1),
                    proj_kk(l, 5, 48, 64),
                    proj_233(l, 64, 96),
                    pool_proj(l, 32, 'avg')
                ], 3, name='concat')
            with tf.variable_scope('incep-35-288a'):
                l = tf.concat([
                    Conv2D('conv11', l, 64, 1),
                    proj_kk(l, 5, 48, 64),
                    proj_233(l, 64, 96),
                    pool_proj(l, 64, 'avg')
                ], 3, name='concat')
            with tf.variable_scope('incep-35-288b'):
                l = tf.concat([
                    Conv2D('conv11', l, 64, 1),
                    proj_kk(l, 5, 48, 64),
                    proj_233(l, 64, 96),
                    pool_proj(l, 64, 'avg')
                ], 3, name='concat')
            # 35x35x288
            with tf.variable_scope('incep-17-768a'):
                l = tf.concat([
                    Conv2D('conv3x3', l, 384, 3, stride=2, padding='VALID'),
                    proj_233(l, 64, 96, stride=2),
                    MaxPooling('maxpool', l, 3, 2)
                ], 3, name='concat')
            with tf.variable_scope('incep-17-768b'):
                l = tf.concat([
                    Conv2D('conv11', l, 192, 1),
                    proj_77(l, 128, 192),
                    proj_277(l, 128, 192),
                    pool_proj(l, 192, 'avg')
                ], 3, name='concat')
            for x in ['c', 'd']:
                with tf.variable_scope('incep-17-768{}'.format(x)):
                    l = tf.concat([
                        Conv2D('conv11', l, 192, 1),
                        proj_77(l, 160, 192),
                        proj_277(l, 160, 192),
                        pool_proj(l, 192, 'avg')
                    ], 3, name='concat')
            with tf.variable_scope('incep-17-768e'):
                l = tf.concat([
                    Conv2D('conv11', l, 192, 1),
                    proj_77(l, 192, 192),
                    proj_277(l, 192, 192),
                    pool_proj(l, 192, 'avg')
                ], 3, name='concat')
            # 17x17x768

            with tf.variable_scope('br1'):
                br1 = AvgPooling('avgpool', l, 5, 3, padding='VALID')
                br1 = Conv2D('conv11', br1, 128, 1)
                shape = br1.get_shape().as_list()
                br1 = Conv2D('convout', br1, 768, shape[1:3], padding='VALID')  # TODO gauss, stddev=0.01
                br1 = FullyConnected('fc', br1, 1000, nl=tf.identity)

            with tf.variable_scope('incep-17-1280a'):
                l = tf.concat([
                    proj_kk(l, 3, 192, 320, stride=2),
                    Conv2D('conv73', proj_77(l, 192, 192), 192, 3, stride=2, padding='VALID'),
                    MaxPooling('maxpool', l, 3, 2)
                ], 3, name='concat')
            for x in ['a', 'b']:
                with tf.variable_scope('incep-8-2048{}'.format(x)) as scope:
                    br11 = Conv2D('conv11', l, 320, 1)
                    br33 = Conv2D('conv133r', l, 384, 1)
                    br33 = tf.concat([
                        Conv2D('conv133a', br33, 384, [1, 3]),
                        Conv2D('conv133b', br33, 384, [3, 1])
                    ], 3, name='conv133')

                    br233 = proj_kk(l, 3, 448, 384)
                    br233 = tf.concat([
                        Conv2D('conv233a', br233, 384, [1, 3]),
                        Conv2D('conv233b', br233, 384, [3, 1]),
                    ], 3, name='conv233')

                    l = tf.concat([
                        br11, br33, br233,
                        pool_proj(l, 192, 'avg')
                    ], 3, name='concat')

            l = GlobalAvgPooling('gap', l)
            # 1x1x2048
            l = Dropout('drop', l, 0.8)
            logits = FullyConnected('linear', l, out_dim=1000, nl=tf.identity)

        loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=br1, labels=label)
        loss1 = tf.reduce_mean(loss1, name='loss1')

        loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss2 = tf.reduce_mean(loss2, name='loss2')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.00004, get_global_step_var(),
                                          80000, 0.7, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='l2_regularize_loss')

        self.cost = tf.add_n([0.4 * loss1, loss2, wd_cost], name='cost')
        add_moving_summary(loss1, loss2, wd_cost, self.cost)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.045, summary=True)
        return tf.train.AdamOptimizer(lr, epsilon=1e-3)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'

    ds = dataset.ILSVRC12(args.data, train_or_test,
                          shuffle=True if isTrain else False, dir_structure='train')
    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_299 = cv2.resize(pp_mean, (299, 299))

    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            def __init__(self):
                self._init(locals())

            def _augment(self, img, _):
                h, w = img.shape[:2]
                size = 299
                scale = self.rng.randint(size, 340) * 1.0 / min(h, w)
                scaleX = scale * self.rng.uniform(0.85, 1.15)
                scaleY = scale * self.rng.uniform(0.85, 1.15)
                desSize = map(int, (max(size, min(w, scaleX * w)),
                                    max(size, min(h, scaleY * h))))
                dst = cv2.resize(img, tuple(desSize), interpolation=cv2.INTER_CUBIC)
                return dst

        augmentors = [
            Resize(),
            imgaug.Rotation(max_deg=10),
            imgaug.RandomApplyAug(imgaug.GaussianBlur(3), 0.5),
            imgaug.Brightness(30, True),
            imgaug.Gamma(),
            imgaug.Contrast((0.8, 1.2), True),
            imgaug.RandomCrop((299, 299)),
            imgaug.RandomApplyAug(imgaug.JpegNoise(), 0.8),
            imgaug.RandomApplyAug(imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (299, 299), 0.2, 3), 0.1),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean_299),
        ]
    else:
        def resize_func(im):
            h, w = im.shape[:2]
            scale = 340.0 / min(h, w)
            desSize = map(int, (max(299, min(w, scale * w)),
                                max(299, min(h, scale * h))))
            im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
            return im
        augmentors = [
            imgaug.MapImage(resize_func),
            imgaug.CenterCrop((299, 299)),
            imgaug.MapImage(lambda x: x - pp_mean_299),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(12, multiprocessing.cpu_count()))
    return ds


def get_config():
    # prepare dataset
    dataset_train = get_data('train')
    dataset_val = get_data('val')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_val, [
                ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(5, 0.03), (9, 0.01), (12, 0.006),
                                       (17, 0.003), (22, 1e-3), (36, 2e-4),
                                       (41, 8e-5), (48, 1e-5), (53, 2e-6)]),
            HumanHyperParamSetter('learning_rate')
        ],
        session_config=get_default_sess_config(0.9),
        model=Model(),
        steps_per_epoch=5000,
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    logger.auto_set_dir()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
