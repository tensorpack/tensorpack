#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: WGAN-mnist.py
# Author: TensorPack contributors

import numpy as np
import tensorflow as tf
import os
import sys
import cv2
import six
import argparse

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.distributions import *
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.gradproc import ScaleGradient, CheckGradient
from GAN import WGANTrainer, WGANModelDesc
"""
To train:
    ./WGAN-mnist.py

To visualize:
    ./WGAN-mnist.py --sample --load path/to/model
"""

BATCH = 64
NOISE_DIM = 100


class MyWGANTrainer(WGANTrainer):

    def _setup(self):
        super(WGANTrainer, self)._setup()
        self.g_updates = 0

    def run_step(self):
        # we use the update strategy from the pytorch implementation by the authors
        d_steps = self.model.critic_runs
        if (self.g_updates < 25) or (self.g_updates % 500 == 0):
            d_steps = 100

        for _ in six.moves.range(d_steps):
            self.sess.run([self.d_min])

        ret = self.sess.run([self.g_min] + self.get_extra_fetches())
        self.g_updates += 1

        return ret[1:]


class Model(WGANModelDesc):

    def get_clipping_weights(self):
        return self.d_vars

    def _get_inputs(self):
        return [InputVar(tf.float32, (None, 28, 28), 'input')]

    def generator(self, z):
        l = FullyConnected('fc0', z, 1024, nl=BNReLU)
        l = FullyConnected('fc1', l, 128 * 7 * 7, nl=BNReLU)
        l = tf.reshape(l, [-1, 7, 7, 128])
        l = Deconv2D('deconv1', l, [14, 14, 64], 4, 2, nl=BNReLU)
        l = Deconv2D('deconv2', l, [28, 28, 1], 4, 2, nl=tf.identity)
        l = tf.tanh(l, name='gen')
        return l

    def discriminator(self, imgs):
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
                argscope(LeakyReLU, alpha=0.1):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', 64)
                 .LeakyReLU()
                 .Conv2D('conv1', 128)
                 .BatchNorm('bn1').LeakyReLU()
                 .FullyConnected('fc1', 1024, nl=tf.identity)
                 .BatchNorm('bn2').LeakyReLU()
                 .FullyConnected('fct', 1, nl=tf.identity)())
        return l

    def _build_graph(self, inputs):
        real_sample = inputs[0]
        real_sample = tf.expand_dims(real_sample * 2.0 - 1, -1)

        z = tf.random_uniform([BATCH, NOISE_DIM], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, NOISE_DIM], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                fake_sample = self.generator(z)
                fake_sample_viz = tf.cast((fake_sample + 1) * 128.0, tf.uint8, name='viz')
                tf.summary.image('gen', fake_sample_viz, max_outputs=30)

            with tf.variable_scope('discrim'):
                real_pred = self.discriminator(real_sample)

            with tf.variable_scope('discrim', reuse=True):
                fake_pred = self.discriminator(fake_sample)

        # wasserstein-GAN objective
        self.build_losses(real_pred, fake_pred)
        summary.add_moving_summary(self.g_loss, self.d_loss)

        # distinguish between variables of generator and discriminator updates
        self.collect_variables()


def get_data():
    ds = ConcatData([dataset.Mnist('train'), dataset.Mnist('test')])
    ds = BatchData(ds, BATCH)
    return ds


def get_config():
    logger.auto_set_dir()
    dataset = get_data()
    lr = symbf.get_scalar_var('learning_rate', 2e-04, summary=True)
    return TrainConfig(
        dataflow=dataset,
        optimizer=tf.train.RMSPropOptimizer(lr),
        callbacks=[ModelSaver()],
        session_config=get_default_sess_config(0.5),
        model=Model(),
        steps_per_epoch=500,
        max_epoch=100,
    )


def sample(model_path):
    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['z'],
        output_names=['gen/viz']))

    while True:
        # only categorical turned on
        z = np.random.uniform(-1, 1, (100, NOISE_DIM))
        o = pred([z])[0]
        viz = next(build_patch_list(o, nr_row=10, nr_col=10))
        viz = cv2.resize(viz, (400, 400))
        interactive_imshow(viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='visualize produced digits by generator')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.sample:
        BATCH = 100
        sample(args.load)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        MyWGANTrainer(config).train()
