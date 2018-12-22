#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: InfoGAN-mnist.py
# Author: Yuxin Wu

import argparse
import numpy as np
import os
import cv2
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import gradproc, optimizer, summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope, under_name_scope
from tensorpack.utils import viz

from GAN import GANModelDesc, GANTrainer

"""
To train:
    ./InfoGAN-mnist.py

To visualize:
    ./InfoGAN-mnist.py --sample --load path/to/model

A pretrained model is at http://models.tensorpack.com/GAN/
"""

BATCH = 128
# latent space is cat(10) x uni(2) x noise(NOISE_DIM)
NUM_CLASS = 10
NUM_UNIFORM = 2
DIST_PARAM_DIM = NUM_CLASS + NUM_UNIFORM
NOISE_DIM = 62
# prior: the assumption how the latent factors are presented in the dataset
DIST_PRIOR_PARAM = [1.] * NUM_CLASS + [0.] * NUM_UNIFORM


def shapeless_placeholder(x, axis, name):
    """
    Make the static shape of a tensor less specific.

    If you want to feed to a tensor, the shape of the feed value must match
    the tensor's static shape. This function creates a placeholder which
    defaults to x if not fed, but has a less specific static shape than x.
    See also `tensorflow#5680 <https://github.com/tensorflow/tensorflow/issues/5680>`_.

    Args:
        x: a tensor
        axis(int or list of ints): these axes of ``x.get_shape()`` will become
            None in the output.
        name(str): name of the output tensor

    Returns:
        a tensor equal to x, but shape information is partially cleared.
    """
    shp = x.get_shape().as_list()
    if not isinstance(axis, list):
        axis = [axis]
    for a in axis:
        if shp[a] is None:
            raise ValueError("Axis {} of shape {} is already unknown!".format(a, shp))
        shp[a] = None
    x = tf.placeholder_with_default(x, shape=shp, name=name)
    return x


def get_distributions(vec_cat, vec_uniform):
    cat = tf.distributions.Categorical(logits=vec_cat, validate_args=True, name='cat')
    uni = tf.distributions.Normal(vec_uniform, scale=1., validate_args=True, allow_nan_stats=False, name='uni_a')
    return cat, uni


def entropy_from_samples(samples, vec):
    """
    Estimate H(x|s) ~= -E_{x \sim P(x|s)}[\log Q(x|s)], where x are samples, and Q is parameterized by vec.
    """
    samples_cat = tf.argmax(samples[:, :NUM_CLASS], axis=1, output_type=tf.int32)
    samples_uniform = samples[:, NUM_CLASS:]
    cat, uniform = get_distributions(vec[:, :NUM_CLASS], vec[:, NUM_CLASS:])

    def neg_logprob(dist, sample, name):
        nll = -dist.log_prob(sample)
        # average over batch
        return tf.reduce_sum(tf.reduce_mean(nll, axis=0), name=name)

    entropies = [neg_logprob(cat, samples_cat, 'nll_cat'),
                 neg_logprob(uniform, samples_uniform, 'nll_uniform')]
    return entropies


@under_name_scope()
def sample_prior(batch_size):
    cat, _ = get_distributions(DIST_PRIOR_PARAM[:NUM_CLASS], DIST_PRIOR_PARAM[NUM_CLASS:])
    sample_cat = tf.one_hot(cat.sample(batch_size), NUM_CLASS)

    """
    OpenAI official code actually models the "uniform" latent code as
    a Gaussian distribution, but obtain the samples from a uniform distribution.
    """
    sample_uni = tf.random_uniform([batch_size, NUM_UNIFORM], -1, 1)
    samples = tf.concat([sample_cat, sample_uni], axis=1)
    return samples


class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 28, 28), 'input')]

    def generator(self, z):
        l = FullyConnected('fc0', z, 1024, activation=BNReLU)
        l = FullyConnected('fc1', l, 128 * 7 * 7, activation=BNReLU)
        l = tf.reshape(l, [-1, 7, 7, 128])
        l = Conv2DTranspose('deconv1', l, 64, 4, 2, activation=BNReLU)
        l = Conv2DTranspose('deconv2', l, 1, 4, 2, activation=tf.identity)
        l = tf.sigmoid(l, name='gen')
        return l

    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        with argscope(Conv2D, kernel_size=4, strides=2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', 64)
                 .tf.nn.leaky_relu()
                 .Conv2D('conv1', 128)
                 .BatchNorm('bn1')
                 .tf.nn.leaky_relu()
                 .FullyConnected('fc1', 1024)
                 .BatchNorm('bn2')
                 .tf.nn.leaky_relu()())

            logits = FullyConnected('fct', l, 1)
            encoder = (LinearWrap(l)
                       .FullyConnected('fce1', 128)
                       .BatchNorm('bne')
                       .tf.nn.leaky_relu()
                       .FullyConnected('fce-out', DIST_PARAM_DIM)())
        return logits, encoder

    def build_graph(self, real_sample):
        real_sample = tf.expand_dims(real_sample, -1)

        # sample the latent code:
        zc = shapeless_placeholder(sample_prior(BATCH), 0, name='z_code')
        z_noise = shapeless_placeholder(
            tf.random_uniform([BATCH, NOISE_DIM], -1, 1), 0, name='z_noise')
        z = tf.concat([zc, z_noise], 1, name='z')

        with argscope([Conv2D, Conv2DTranspose, FullyConnected],
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                fake_sample = self.generator(z)
                fake_sample_viz = tf.cast((fake_sample) * 255.0, tf.uint8, name='viz')
                tf.summary.image('gen', fake_sample_viz, max_outputs=30)

            # may need to investigate how bn stats should be updated across two discrim
            with tf.variable_scope('discrim'):
                real_pred, _ = self.discriminator(real_sample)
                fake_pred, dist_param = self.discriminator(fake_sample)

        """
        Mutual information between x (i.e. zc in this case) and some
        information s (the generated samples in this case):

                    I(x;s) = H(x) - H(x|s)
                           = H(x) + E[\log P(x|s)]

        The distribution from which zc is sampled, in this case, is set to a fixed prior already.
        So the first term is a constant.
        For the second term, we can maximize its variational lower bound:
                    E_{x \sim P(x|s)}[\log Q(x|s)]
        where Q(x|s) is a proposal distribution to approximate P(x|s).

        Here, Q(x|s) is assumed to be a distribution which shares the form
        of P, and whose parameters are predicted by the discriminator network.
        """
        with tf.name_scope("mutual_information"):
            with tf.name_scope('prior_entropy'):
                cat, uni = get_distributions(DIST_PRIOR_PARAM[:NUM_CLASS], DIST_PRIOR_PARAM[NUM_CLASS:])
                ents = [cat.entropy(name='cat_entropy'), tf.reduce_sum(uni.entropy(), name='uni_entropy')]
                entropy = tf.add_n(ents, name='total_entropy')
                # Note that the entropy of prior is a constant. The paper mentioned it but didn't use it.

            with tf.name_scope('conditional_entropy'):
                cond_ents = entropy_from_samples(zc, dist_param)
                cond_entropy = tf.add_n(cond_ents, name="total_entropy")

            MI = tf.subtract(entropy, cond_entropy, name='mutual_information')
            summary.add_moving_summary(entropy, cond_entropy, MI, *cond_ents)

        # default GAN objective
        self.build_losses(real_pred, fake_pred)

        # subtract mutual information for latent factors (we want to maximize them)
        self.g_loss = tf.subtract(self.g_loss, MI, name='total_g_loss')
        self.d_loss = tf.subtract(self.d_loss, MI, name='total_d_loss')

        summary.add_moving_summary(self.g_loss, self.d_loss)

        # distinguish between variables of generator and discriminator updates
        self.collect_variables()

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, dtype=tf.float32, trainable=False)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-6)
        # generator learns 5 times faster
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(('gen/.*', 5))])


def get_data():
    ds = ConcatData([dataset.Mnist('train'), dataset.Mnist('test')])
    ds = BatchData(ds, BATCH)
    ds = MapData(ds, lambda dp: [dp[0]])  # only use the image
    return ds


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
        viz1 = viz.stack_patches(o, nr_row=10, nr_col=10)
        viz1 = cv2.resize(viz1, (IMG_SIZE, IMG_SIZE))

        # show effect of first continous variable with fixed noise
        zc = np.concatenate((z_cat, z_uni, z_uni * 0), axis=1)
        o = pred(zc, z_noise * 0)[0]
        viz2 = viz.stack_patches(o, nr_row=10, nr_col=10)
        viz2 = cv2.resize(viz2, (IMG_SIZE, IMG_SIZE))

        # show effect of second continous variable with fixed noise
        zc = np.concatenate((z_cat, z_uni * 0, z_uni), axis=1)
        o = pred(zc, z_noise * 0)[0]
        viz3 = viz.stack_patches(o, nr_row=10, nr_col=10)
        viz3 = cv2.resize(viz3, (IMG_SIZE, IMG_SIZE))

        canvas = viz.stack_patches(
            [viz1, viz2, viz3],
            nr_row=1, nr_col=3, border=5, bgcolor=(255, 0, 0))

        viz.interactive_imshow(canvas)


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
        logger.auto_set_dir()
        GANTrainer(QueueInput(get_data()),
                   Model()).train_with_defaults(
            callbacks=[ModelSaver(keep_checkpoint_every_n_hours=0.1)],
            steps_per_epoch=500,
            max_epoch=100,
            session_init=SaverRestore(args.load) if args.load else None
        )
