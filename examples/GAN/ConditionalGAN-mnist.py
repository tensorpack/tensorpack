#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ConditionalGAN-mnist.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import os
import cv2
import argparse


from tensorpack import *
from tensorpack.utils.viz import interactive_imshow, stack_patches
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.dataflow import dataset
from GAN import GANTrainer, RandomZData, GANModelDesc

"""
To train:
    ./ConditionalGAN-mnist.py

To visualize:
    ./ConditionalGAN-mnist.py --sample --load path/to/model

A pretrained model is at http://models.tensorpack.com/GAN/
"""

BATCH = 128


class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 28, 28), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def generator(self, z, y):
        l = FullyConnected('fc0', tf.concat([z, y], 1), 1024, nl=BNReLU)
        l = FullyConnected('fc1', tf.concat([l, y], 1), 64 * 2 * 7 * 7, nl=BNReLU)
        l = tf.reshape(l, [-1, 7, 7, 64 * 2])

        y = tf.reshape(y, [-1, 1, 1, 10])
        l = tf.concat([l, tf.tile(y, [1, 7, 7, 1])], 3)
        l = Deconv2D('deconv1', l, 64 * 2, 5, 2, nl=BNReLU)

        l = tf.concat([l, tf.tile(y, [1, 14, 14, 1])], 3)
        l = Deconv2D('deconv2', l, 1, 5, 2, nl=tf.identity)
        l = tf.nn.tanh(l, name='gen')
        return l

    @auto_reuse_variable_scope
    def discriminator(self, imgs, y):
        """ return a (b, 1) logits"""
        yv = y
        y = tf.reshape(y, [-1, 1, 1, 10])
        with argscope(Conv2D, nl=tf.identity, kernel_shape=5, stride=2), \
                argscope(LeakyReLU, alpha=0.2):
            l = (LinearWrap(imgs)
                 .ConcatWith(tf.tile(y, [1, 28, 28, 1]), 3)
                 .Conv2D('conv0', 11)
                 .LeakyReLU()

                 .ConcatWith(tf.tile(y, [1, 14, 14, 1]), 3)
                 .Conv2D('conv1', 74)
                 .BatchNorm('bn1').LeakyReLU()

                 .apply(symbf.batch_flatten)
                 .ConcatWith(yv, 1)
                 .FullyConnected('fc1', 1024, nl=tf.identity)
                 .BatchNorm('bn2').LeakyReLU()

                 .ConcatWith(yv, 1)
                 .FullyConnected('fct', 1, nl=tf.identity)())
        return l

    def _build_graph(self, inputs):
        image_pos, y = inputs
        image_pos = tf.expand_dims(image_pos * 2.0 - 1, -1)
        y = tf.one_hot(y, 10, name='label_onehot')

        z = tf.random_uniform([BATCH, 100], -1, 1, name='z_train')
        z = symbf.shapeless_placeholder(z, [0], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z, y)
                tf.summary.image('gen', image_gen, 30)
            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos, y)
                vecneg = self.discriminator(image_gen, y)

        self.build_losses(vecpos, vecneg)
        self.collect_variables()

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(2e-4, beta1=0.5, epsilon=1e-3)


def get_data():
    ds = ConcatData([dataset.Mnist('train'), dataset.Mnist('test')])
    return BatchData(ds, BATCH)


def sample(model_path):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['label', 'z'],
        output_names=['gen/gen'])

    ds = MapData(RandomZData((100, 100)),
                 lambda dp: [np.arange(100) % 10, dp[0]])
    pred = SimpleDatasetPredictor(pred, ds)
    for o in pred.get_result():
        o = o[0] * 255.0
        viz = stack_patches(o, nr_row=10, nr_col=10)
        viz = cv2.resize(viz, (800, 800))
        interactive_imshow(viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.sample:
        sample(args.load)
    else:
        logger.auto_set_dir()
        GANTrainer(QueueInput(get_data()), Model()).train_with_defaults(
            callbacks=[ModelSaver()],
            steps_per_epoch=500,
            max_epoch=100,
            session_init=SaverRestore(args.load) if args.load else None
        )
