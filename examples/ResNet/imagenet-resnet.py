#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
Training code of Pre-Activation version of ResNet on ImageNet. Work In Progress.
Top1 error is now about 0.5% higher than fb.resnet.torch.
"""


NR_GPU = 4
TOTAL_BATCH_SIZE = 256
BATCH_SIZE = TOTAL_BATCH_SIZE / NR_GPU
INPUT_SHAPE = 224

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars):
        image, label = input_vars

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[-1]
            input = l
            if preact == 'both_preact':
                l = BatchNorm('preact', l)
                l = tf.nn.relu(l, name='preact-relu')
                input = l
            elif preact != 'no_preact':
                l = BatchNorm('preact', l)
                l = tf.nn.relu(l, name='preact-relu')
            l = Conv2D('conv1', l, ch_out, 3, stride=stride)
            l = BatchNorm('bn', l)
            l = tf.nn.relu(l)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[-1]
            input = l
            if preact == 'both_preact':
                l = BatchNorm('preact', l)
                l = tf.nn.relu(l, name='preact-relu')
                input = l
            elif preact != 'no_preact':
                l = BatchNorm('preact', l)
                l = tf.nn.relu(l, name='preact-relu')
            l = Conv2D('conv1', l, ch_out, 1)
            l = BatchNorm('bn1', l)
            l = tf.nn.relu(l)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride)
            l = BatchNorm('bn2', l)
            l = tf.nn.relu(l)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                            'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        cfg = {
            18: ([2,2,2,2], basicblock),
            34: ([3,4,6,3], basicblock),
            50: ([3,4,6,3], bottleneck),
            101: ([3,4,23,3], bottleneck)
        }
        defs, block_func = cfg[50]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                W_init=variance_scaling_initializer(mode='FAN_OUT')):
            logits = (LinearWrap(image)
                .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                .apply(layer, 'group1', block_func, 128, defs[1], 2)
                .apply(layer, 'group2', block_func, 256, defs[2], 2)
                .apply(layer, 'group3', block_func, 512, defs[3], 2)
                .BatchNorm('bnlast')
                .tf.nn.relu()
                .GlobalAvgPooling('gap')
                .FullyConnected('linear', 1000, nl=tf.identity)())

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        wrong = prediction_incorrect(logits, label, 1)
        nr_wrong = tf.reduce_sum(wrong, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5)
        nr_wrong = tf.reduce_sum(wrong, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(1e-4, get_global_step_var(),
                                          200000, 0.7, True)
        wd_cost = tf.mul(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)

        self.cost = tf.add_n([loss, wd_cost], name='cost')

def get_data(train_or_test):
    isTrain = train_or_test == 'train'

    datadir = args.data
    ds = dataset.ILSVRC12(datadir, train_or_test,
            shuffle=True if isTrain else False, dir_structure='original')
    image_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    image_std = np.array([0.229, 0.224, 0.225], dtype='float32')

    if isTrain:
        def resize_func(img):
            # crop 8%~100% of the original image
            # See `Going Deeper with Convolutions` by Google.
            h, w = img.shape[:2]
            area = h * w
            for _ in range(10):
                targetArea = self.rng.uniform(0.08, 1.0) * area
                aspectR = self.rng.uniform(0.75,1.333)
                ww = int(np.sqrt(targetArea * aspectR))
                hh = int(np.sqrt(targetArea / aspectR))
                if self.rng.uniform() < 0.5:
                    ww, hh = hh, ww
                if hh <= h and ww <= w:
                    x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                    y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                    out = img[y1:y1+hh,x1:x1+ww]
                    out = cv2.resize(out, (224,224), interpolation=cv2.INTER_CUBIC)
                    return out
            out = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
            return out

        augmentors = [
            imgaug.MapImage(resize_func),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 imgaug.Lighting(0.1,
                     eigval=[0.2175, 0.0188, 0.0045],
                     eigvec=[[ -0.5675,  0.7192,  0.4009],
                      [ -0.5808, -0.0045, -0.8140],
                      [ -0.5836, -0.6948,  0.4203]]
                 )]),
            imgaug.Clip(),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: (x * (1.0 / 255) - image_mean) / image_std),
        ]
    else:
        def resize_func(im):
            h, w = im.shape[:2]
            scale = 256.0 / min(h, w)
            desSize = map(int, [scale * w, scale * h])
            im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
            return im
        augmentors = [
            imgaug.MapImage(resize_func),
            imgaug.CenterCrop((224, 224)),
            imgaug.MapImage(lambda x: (x * (1.0 / 255) - image_mean) / image_std),
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

    sess_config = get_default_sess_config(0.99)

    lr = tf.Variable(0.1, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            InferenceRunner(dataset_val, [
                ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(30, 1e-2), (60, 1e-3), (85, 2e-4)]),
            HumanHyperParamSetter('learning_rate'),
        ]),
        session_config=sess_config,
        model=Model(),
        step_per_epoch=5000,
        max_epoch=110,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
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
