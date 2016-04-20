#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn_resnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

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
from tensorpack.dataflow import imgaug


"""
ResNet-110 for SVHN Digit Classification.
Reach 1.8% validation error after 70 epochs, with 2 TitanX. 2it/s.
You might need to adjust the learning rate schedule when running with 1 GPU.
"""

BATCH_SIZE = 128

class Model(ModelDesc):
    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 32, 32, 3], 'input'),
                InputVar(tf.int32, [None], 'label')
               ]

    def _get_cost(self, input_vars, is_training):
        image, label = input_vars
        image = image / 128.0 - 1

        def conv(name, l, channel, stride):
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[3]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name) as scope:
                if not first:
                    b1 = BatchNorm('bn1', l, is_training)
                    b1 = tf.nn.relu(b1)
                else:
                    b1 = l
                c1 = conv('conv1', b1, out_channel, stride1)
                b2 = BatchNorm('bn2', c1, is_training)
                b2 = tf.nn.relu(b2)
                c2 = conv('conv2', b2, out_channel, 1)

                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0,0], [0,0], [0,0], [in_channel//2, in_channel//2]])

                l = c2 + l
                return l


        l = conv('conv0', image, 16, 1)
        l = BatchNorm('bn0', l, is_training)
        l = tf.nn.relu(l)
        l = residual('res1.0', l, first=True)
        for k in range(1, self.n):
            l = residual('res1.{}'.format(k), l)
        # 32,c=16

        l = residual('res2.0', l, increase_dim=True)
        for k in range(1, self.n):
            l = residual('res2.{}'.format(k), l)
        # 16,c=32

        l = residual('res3.0', l, increase_dim=True)
        for k in range(1, self.n):
            l = residual('res3.' + str(k), l)
        l = BatchNorm('bnlast', l, is_training)
        l = tf.nn.relu(l)
        # 8,c=64
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost)

        wrong = prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        tf.add_to_collection(
            MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        #wd_cost = regularize_cost('.*/W', l2_regularizer(0.0002), name='regularize_loss')
        wd_w = tf.train.exponential_decay(0.0001, get_global_step_var(),
                                          960000, 0.5, True)
        wd_cost = wd_w * regularize_cost('.*/W', tf.nn.l2_loss)
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    pp_mean = dataset.SVHNDigit.get_per_pixel_mean()
    if isTrain:
        d1 = dataset.SVHNDigit('train')
        d2 = dataset.SVHNDigit('extra')
        ds = RandomMixData([d1, d2])
    else:
        ds = dataset.SVHNDigit('test')

    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            #imgaug.Flip(horiz=True),
            imgaug.BrightnessAdd(10),
            imgaug.Contrast((0.8,1.2)),
            imgaug.GaussianDeform(  # this is slow
                [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
                (32, 32), 0.2, 3),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 5, 5)
    return ds

def get_config():
    # prepare dataset
    dataset_train = get_data('train')
    step_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    sess_config = get_default_sess_config(0.9)

    lr = tf.Variable(0.1, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(dataset_test,
                [ScalarStats('cost'), ClassificationError() ]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (20, 0.01), (28, 0.001), (50, 0.0001)])
        ]),
        session_config=sess_config,
        model=Model(n=18),
        step_per_epoch=step_per_epoch,
        max_epoch=500,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    basename = os.path.basename(__file__)
    logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        QueueInputTrainer(config).train()
