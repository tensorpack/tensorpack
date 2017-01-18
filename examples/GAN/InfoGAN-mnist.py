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

    def generator(self, z):
        l = FullyConnected('fc0', z, 1024, nl=BNReLU)
        l = FullyConnected('fc1', l, 128 * 7 * 7, nl=BNReLU)
        l = tf.reshape(l, [-1, 7, 7, 128])
        l = Deconv2D('deconv1', l, [14, 14, 64], 4, 2, nl=BNReLU)
        l = Deconv2D('deconv2', l, [28, 28, 1], 4, 2, nl=tf.identity)
        l = tf.nn.tanh(l, name='gen')
        return l

    def discriminator(self, imgs):
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
                       .FullyConnected('fce-out', self.factors.param_dim(), nl=tf.identity)())
        return logits, encoder

    def _build_graph(self, input_vars):

        real_sample = input_vars[0]
        real_sample = tf.expand_dims(real_sample * 2.0 - 1, -1)

        # latent space is cat(10) x uni(1) x uni(1) x noise(62)
        self.factors = ProductDistribution("factors", [CategoricalDistribution("cat", 10),
                                                       UniformDistribution("uni_a", 1),
                                                       UniformDistribution("uni_b", 1),
                                                       NoiseDistribution("noise", 62)])

        z = self.factors.sample(BATCH)

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                fake_sample = self.generator(z)
                tf.summary.image('gen', fake_sample, max_outputs=30)

            with tf.variable_scope('discrim'):
                real_pred, _ = self.discriminator(real_sample)

            with tf.variable_scope('discrim', reuse=True):
                fake_pred, dist_param = self.discriminator(fake_sample)

        # post-process all dist_params from discriminator
        encoder_activation = self.factors.encoder_activation(dist_param)

        with tf.name_scope("mutual_information"):
            MIs = self.factors.mutual_information(z, encoder_activation)
            mi = tf.add_n(MIs, name="total")
        summary.add_moving_summary(MIs + [mi])

        # default GAN objective
        self.build_losses(real_pred, fake_pred)

        # subtract mutual information for latent factores (we want to maximize them)
        self.g_loss = tf.subtract(self.g_loss, mi, name='total_g_loss')
        self.d_loss = tf.subtract(self.d_loss, mi, name='total_d_loss')

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
        input_names=['z_cat', 'z_uni_a', 'z_uni_b', 'z_noise'],
        output_names=['gen/gen']))

    # sample all one-hot encodings (10 times)
    z_cat = np.tile(np.eye(10), [10, 1])
    # sample continuos variables from -2 to +2 as mentioned in the paper
    z_uni = np.linspace(-2.0, 2.0, num=100)
    z_uni = z_uni[:, None]

    IMG_SIZE = 400

    while True:
        # only categorical turned on
        z_noise = np.random.uniform(-1, 1, (100, 88))
        o = pred([z_cat, z_uni * 0, z_uni * 0, z_noise])
        o = (o[0] + 1) * 128.0
        viz1 = next(build_patch_list(o, nr_row=10, nr_col=10))
        viz1 = cv2.resize(viz1, (IMG_SIZE, IMG_SIZE))

        # show effect of first continous variable with fixed noise
        o = pred([z_cat, z_uni, z_uni * 0, z_noise * 0])
        o = (o[0] + 1) * 128.0
        viz2 = next(build_patch_list(o, nr_row=10, nr_col=10))
        viz2 = cv2.resize(viz2, (IMG_SIZE, IMG_SIZE))

        # show effect of second continous variable with fixed noise
        o = pred([z_cat, z_uni * 0, z_uni, z_noise * 0])
        o = (o[0] + 1) * 128.0
        viz3 = next(build_patch_list(o, nr_row=10, nr_col=10))
        viz3 = cv2.resize(viz3, (IMG_SIZE, IMG_SIZE))

        viz = stack_images([viz1, viz2, viz3])

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
        BATCH = 100
        sample(args.load)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        GANTrainer(config).train()
