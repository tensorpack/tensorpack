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
from tensorpack.tfutils.distributions import *
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.gradproc import ScaleGradient, CheckGradient
from GAN import GANTrainer, GANModelDesc

"""
To train:
    ./InfoGAN-mnist.py

To visualize:
    ./InfoGAN-mnist.py --sample --load path/to/model

A pretrained model is at https://drive.google.com/open?id=0B9IPQTvr2BBkLUF2M0RXU1NYSkE
"""

BATCH = 128
NOISE_DIM = 62


class GaussianWithUniformSample(GaussianDistribution):
    """
    OpenAI official code actually models the "uniform" latent code as
    a Gaussian distribution, but obtain the samples from a uniform distribution.
    We follow the official code for now.
    """
    def _sample(self, batch_size, theta):
        return tf.random_uniform([batch_size, self.dim], -1, 1)


class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 28, 28), 'input')]

    def generator(self, z):
        l = FullyConnected('fc0', z, 1024, nl=BNReLU)
        l = FullyConnected('fc1', l, 128 * 7 * 7, nl=BNReLU)
        l = tf.reshape(l, [-1, 7, 7, 128])
        l = Deconv2D('deconv1', l, [14, 14, 64], 4, 2, nl=BNReLU)
        l = Deconv2D('deconv2', l, [28, 28, 1], 4, 2, nl=tf.identity)
        l = tf.sigmoid(l, name='gen')
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
                       .FullyConnected('fce-out', self.factors.param_dim, nl=tf.identity)())
        return logits, encoder

    def _build_graph(self, inputs):
        real_sample = inputs[0]
        real_sample = tf.expand_dims(real_sample, -1)

        # latent space is cat(10) x uni(1) x uni(1) x noise(NOISE_DIM)
        self.factors = ProductDistribution("factors", [CategoricalDistribution("cat", 10),
                                                       GaussianWithUniformSample("uni_a", 1),
                                                       GaussianWithUniformSample("uni_b", 1)])
        # prior: the assumption how the factors are presented in the dataset
        prior = tf.constant([0.1] * 10 + [0, 0], tf.float32, [12], name='prior')
        batch_prior = tf.tile(tf.expand_dims(prior, 0), [BATCH, 1], name='batch_prior')

        # sample the latent code:
        zc = symbf.shapeless_placeholder(
            self.factors.sample(BATCH, prior), 0, name='z_code')
        z_noise = symbf.shapeless_placeholder(
            tf.random_uniform([BATCH, NOISE_DIM], -1, 1), 0, name='z_noise')
        z = tf.concat([zc, z_noise], 1, name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                fake_sample = self.generator(z)
                fake_sample_viz = tf.cast((fake_sample) * 255.0, tf.uint8, name='viz')
                tf.summary.image('gen', fake_sample_viz, max_outputs=30)

            # may need to investigate how bn stats should be updated across two discrim
            with tf.variable_scope('discrim'):
                real_pred, _ = self.discriminator(real_sample)

            with tf.variable_scope('discrim', reuse=True):
                fake_pred, dist_param = self.discriminator(fake_sample)

        """
        Mutual information between x (i.e. zc in this case) and some
        information s (the generated samples in this case):

                    I(x;s) = H(x) - H(x|s)
                           = H(x) + E[\log P(x|s)]

        The distribution from which zc is sampled, in this case, is set to a fixed prior already.
        For the second term, we can maximize its variational lower bound:
                    E_{x \sim P(x|s)}[\log Q(x|s)]
        where Q(x|s) is a proposal distribution to approximate P(x|s).

        Here, Q(x|s) is assumed to be a distribution which shares the form
        of self.factors, and whose parameters are predicted by the discriminator network.
        """
        with tf.name_scope("mutual_information"):
            ents = self.factors.entropy(zc, batch_prior)
            entropy = tf.add_n(ents, name='total_entropy')
            # Note that dropping this term has no effect because the entropy
            # of prior is a constant. The paper mentioned it but didn't use it.
            # Adding this term may make the curve less stable because the
            # entropy estimated from the samples is not the true value.

            # post-process output vector from discriminator to obtain valid distribution parameters
            encoder_activation = self.factors.encoder_activation(dist_param)
            cond_ents = self.factors.entropy(zc, encoder_activation)
            cond_entropy = tf.add_n(cond_ents, name="total_conditional_entropy")

            MI = tf.subtract(entropy, cond_entropy, name='mutual_information')
            summary.add_moving_summary(entropy, cond_entropy, MI, *ents)

        # default GAN objective
        self.build_losses(real_pred, fake_pred)

        # subtract mutual information for latent factors (we want to maximize them)
        self.g_loss = tf.subtract(self.g_loss, MI, name='total_g_loss')
        self.d_loss = tf.subtract(self.d_loss, MI, name='total_d_loss')

        summary.add_moving_summary(self.g_loss, self.d_loss)

        # distinguish between variables of generator and discriminator updates
        self.collect_variables()

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 2e-4, summary=True)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-6)
        # generator learns 5 times faster
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(('gen/.*', 5), log=True)])


def get_data():
    ds = ConcatData([dataset.Mnist('train'), dataset.Mnist('test')])
    ds = BatchData(ds, BATCH)
    return ds


def get_config():
    logger.auto_set_dir()
    return TrainConfig(
        dataflow=get_data(),
        callbacks=[ModelSaver(keep_freq=0.1)],
        session_config=get_default_sess_config(0.5),
        model=Model(),
        steps_per_epoch=500,
        max_epoch=100,
    )


def sample(model_path):
    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['z_code', 'z_noise'],
        output_names=['gen/viz']))

    # sample all one-hot encodings (10 times)
    z_cat = np.tile(np.eye(10), [10, 1])
    # sample continuos variables from -2 to +2 as mentioned in the paper
    z_uni = np.linspace(-2.0, 2.0, num=100)
    z_uni = z_uni[:, None]

    IMG_SIZE = 400

    while True:
        # only categorical turned on
        z_noise = np.random.uniform(-1, 1, (100, NOISE_DIM))
        zc = np.concatenate((z_cat, z_uni * 0, z_uni * 0), axis=1)
        o = pred(zc, z_noise)[0]
        viz1 = stack_patches(o, nr_row=10, nr_col=10)
        viz1 = cv2.resize(viz1, (IMG_SIZE, IMG_SIZE))

        # show effect of first continous variable with fixed noise
        zc = np.concatenate((z_cat, z_uni, z_uni * 0), axis=1)
        o = pred(zc, z_noise * 0)[0]
        viz2 = stack_patches(o, nr_row=10, nr_col=10)
        viz2 = cv2.resize(viz2, (IMG_SIZE, IMG_SIZE))

        # show effect of second continous variable with fixed noise
        zc = np.concatenate((z_cat, z_uni * 0, z_uni), axis=1)
        o = pred(zc, z_noise * 0)[0]
        viz3 = stack_patches(o, nr_row=10, nr_col=10)
        viz3 = cv2.resize(viz3, (IMG_SIZE, IMG_SIZE))

        viz = stack_patches(
            [viz1, viz2, viz3],
            nr_row=1, nr_col=3, border=5, bgcolor=(255, 0, 0))

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
