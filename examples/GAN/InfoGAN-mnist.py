#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: InfoGAN-mnist.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import os
import sys
import cv2
import argparse

from tensorpack import *
from tensorpack.utils.viz import *
import tensorpack.tfutils.symbolic_functions as symbf
from GAN import GANTrainer, GANModelDesc

BATCH = 128


class Model(GANModelDesc):

    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, 28, 28), 'input')]

    def generator(self, z, c):
        z = tf.concat_v2([c, z], 1, name='fullz')

        l = FullyConnected('fc0', z, 1024, nl=BNReLU)
        l = FullyConnected('fc1', l, 128 * 7 * 7, nl=BNReLU)
        l = tf.reshape(l, [-1, 7, 7, 128])
        l = Deconv2D('deconv1', l, [14, 14, 64], 4, 2, nl=BNReLU)
        l = Deconv2D('deconv2', l, [28, 28, 1], 4, 2, nl=tf.identity)
        l = tf.nn.tanh(l, name='gen')
        return l

    def discriminator(self, imgs):
        """ return a (b, 1) logits"""
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
                argscope(LeakyReLU, alpha=0.2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', 64)
                 .LeakyReLU()
                 .Conv2D('conv1', 128)
                 .BatchNorm('bn1').LeakyReLU()
                 .FullyConnected('fc1', 1024, nl=tf.identity)
                 .BatchNorm('bn2').LeakyReLU()())

            logits = FullyConnected('fct', l, 1, nl=tf.identity)
            encoder = (LinearWrap(l)
                       .FullyConnected('fce1', 128, nl=tf.identity)
                       .BatchNorm('bne').LeakyReLU()
                       .FullyConnected('fce-out', 10, nl=tf.identity)())
        return logits, encoder

    def _build_graph(self, input_vars):

        latent_factor = CategoricalDistribution("cat", 10)

        real_sample = input_vars[0]
        real_sample = tf.expand_dims(real_sample * 2.0 - 1, -1)

        zc = latent_factor.code(BATCH, name='zc')

        z = tf.random_uniform(tf.stack([tf.shape(zc)[0], 90]), -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, 90], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                fake_sample = self.generator(z, zc)
                tf.summary.image('gen', fake_sample, max_outputs=30)

            with tf.variable_scope('discrim'):
                real_pred, _ = self.discriminator(real_sample)

            with tf.variable_scope('discrim', reuse=True):
                fake_pred, dist_param = self.discriminator(fake_sample)
                prob = tf.nn.softmax(dist_param)  # log prob of each category

        Hc = latent_factor.entropy(zc)
        MIloss = latent_factor.mutual_information(zc, prob)

        self.build_losses(real_pred, fake_pred)
        self.g_loss = tf.subtract(self.g_loss, MIloss, name='total_g_loss')
        self.d_loss = tf.subtract(self.d_loss, MIloss, name='total_d_loss')
        summary.add_moving_summary(MIloss, self.g_loss, self.d_loss, Hc)

        self.collect_variables()


def get_data():
    ds = ConcatData([dataset.Mnist('train'), dataset.Mnist('test')])
    ds = BatchData(ds, BATCH)
    return ds


def get_config():
    logger.auto_set_dir()
    dataset = get_data()
    lr = symbf.get_scalar_var('learning_rate', 2e-4, summary=True)
    return TrainConfig(
        dataflow=dataset,
        optimizer=tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
        ]),
        session_config=get_default_sess_config(0.5),
        model=Model(),
        step_per_epoch=500,
        max_epoch=100,
    )


def sample(model_path):
    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['zc'],
        output_names=['gen/gen']))

    eye = []
    for k in np.eye(10):
        eye = eye + [k] * 10
    inputs = np.asarray(eye)
    while True:
        o = pred([inputs])
        o = (o[0] + 1) * 128.0
        viz = next(build_patch_list(o, nr_row=10, nr_col=10))
        viz = cv2.resize(viz, (800, 800))
        interactive_imshow(viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='visualize the space of the 10 latent codes')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.sample:
        sample(args.load)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        GANTrainer(config).train()
