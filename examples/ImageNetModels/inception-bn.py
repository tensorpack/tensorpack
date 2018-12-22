#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inception-bn.py
# Author: Yuxin Wu

import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import fbresnet_augmentor, get_imagenet_dataflow

# Change them if using different number of GPUs.
TOTAL_BATCH_SIZE = 64 * 6
NUM_GPU = 6
BATCH_SIZE = TOTAL_BATCH_SIZE // NUM_GPU
INPUT_SHAPE = 224


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = image / 128.0

        def inception(name, x, nr1x1, nr3x3r, nr3x3, nr233r, nr233, nrpool, pooltype):
            stride = 2 if nr1x1 == 0 else 1
            with tf.variable_scope(name):
                outs = []
                if nr1x1 != 0:
                    outs.append(Conv2D('conv1x1', x, nr1x1, 1))
                x2 = Conv2D('conv3x3r', x, nr3x3r, 1)
                outs.append(Conv2D('conv3x3', x2, nr3x3, 3, strides=stride))

                x3 = Conv2D('conv233r', x, nr233r, 1)
                x3 = Conv2D('conv233a', x3, nr233, 3)
                outs.append(Conv2D('conv233b', x3, nr233, 3, strides=stride))

                if pooltype == 'max':
                    x4 = MaxPooling('mpool', x, 3, stride, padding='SAME')
                else:
                    assert pooltype == 'avg'
                    x4 = AvgPooling('apool', x, 3, stride, padding='SAME')
                if nrpool != 0:  # pool + passthrough if nrpool == 0
                    x4 = Conv2D('poolproj', x4, nrpool, 1)
                outs.append(x4)
                return tf.concat(outs, 3, name='concat')

        with argscope(Conv2D, activation=BNReLU, use_bias=False):
            l = (LinearWrap(image)
                 .Conv2D('conv0', 64, 7, strides=2)
                 .MaxPooling('pool0', 3, 2, padding='SAME')
                 .Conv2D('conv1', 64, 1)
                 .Conv2D('conv2', 192, 3)
                 .MaxPooling('pool2', 3, 2, padding='SAME')())
            # 28
            l = inception('incep3a', l, 64, 64, 64, 64, 96, 32, 'avg')
            l = inception('incep3b', l, 64, 64, 96, 64, 96, 64, 'avg')
            l = inception('incep3c', l, 0, 128, 160, 64, 96, 0, 'max')

            br1 = (LinearWrap(l)
                   .Conv2D('loss1conv', 128, 1)
                   .FullyConnected('loss1fc', 1024, activation=tf.nn.relu)
                   .FullyConnected('loss1logit', 1000, activation=tf.identity)())
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=br1, labels=label)
            loss1 = tf.reduce_mean(loss1, name='loss1')

            # 14
            l = inception('incep4a', l, 224, 64, 96, 96, 128, 128, 'avg')
            l = inception('incep4b', l, 192, 96, 128, 96, 128, 128, 'avg')
            l = inception('incep4c', l, 160, 128, 160, 128, 160, 128, 'avg')
            l = inception('incep4d', l, 96, 128, 192, 160, 192, 128, 'avg')
            l = inception('incep4e', l, 0, 128, 192, 192, 256, 0, 'max')

            br2 = Conv2D('loss2conv', l, 128, 1)
            br2 = FullyConnected('loss2fc', br2, 1024, activation=tf.nn.relu)
            br2 = FullyConnected('loss2logit', br2, 1000, activation=tf.identity)
            loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=br2, labels=label)
            loss2 = tf.reduce_mean(loss2, name='loss2')

            # 7
            l = inception('incep5a', l, 352, 192, 320, 160, 224, 128, 'avg')
            l = inception('incep5b', l, 352, 192, 320, 192, 224, 128, 'max')
            l = GlobalAvgPooling('gap', l)

            logits = FullyConnected('linear', l, 1000, activation=tf.identity)
        tf.nn.softmax(logits, name='output')
        loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss3 = tf.reduce_mean(loss3, name='loss3')

        cost = tf.add_n([loss3, 0.3 * loss2, 0.3 * loss1], name='weighted_cost')
        add_moving_summary(cost, loss1, loss2, loss3)

        def prediction_incorrect(logits, label, topk, name):
            return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)), tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train_error_top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train_error_top5'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          80000, 0.7, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='l2_regularize_loss')

        total_cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(wd_cost, total_cost)
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.045, trainable=False)
        return tf.train.MomentumOptimizer(lr, 0.9)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    augs = fbresnet_augmentor(isTrain)

    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    augs.append(imgaug.MapImage(lambda x: x - pp_mean[16:-16, 16:-16]))

    ds = get_imagenet_dataflow(args.data, train_or_test, BATCH_SIZE, augs)
    return ds


def get_config():
    logger.auto_set_dir()
    dataset_train = get_data('train')
    dataset_val = get_data('val')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_val, [
                ClassificationError('wrong-top1', 'val-top1-error'),
                ClassificationError('wrong-top5', 'val-top5-error')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(8, 0.03), (14, 0.02), (17, 5e-3),
                                       (19, 3e-3), (24, 1e-3), (26, 2e-4),
                                       (30, 5e-5)])
        ],
        model=Model(),
        steps_per_epoch=5000,
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
    nr_tower = get_num_gpu()
    assert nr_tower == NUM_GPU
    launch_train_with_config(config, SyncMultiGPUTrainer(NUM_GPU))
