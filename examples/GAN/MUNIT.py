#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: MUNIT.py
# Author: Aaron Gokaslan

import os
import argparse
import glob
from functools import partial
from six.moves import range

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.gpu import get_num_gpu
import tensorflow as tf
from GAN import GANTrainer, GANModelDesc

"""
MUNIT.py
Training and testing visualizations will be in tensorboard.
"""

SHAPE = 256
BATCH = 1
TEST_BATCH = 8
NF = 64  # channel size
N_DIS = 4
N_SAMPLE = 2
N_SCALE = 3
N_RES = 4
STYLE_DIM = 8
USE_CYCLE = False
REGULARIZE = True
enable_argscope_for_module(tf.layers)


def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)


def tpad(x, pad, mode='CONSTANT', name=None):
    return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=mode)


def AdaIN(x, gamma=1.0, beta=0, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP

    mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='AdaIn')


def AdaINReLU(x, gamma=1.0, beta=0.0, name=None):
    x = AdaIN(x, gamma=gamma, beta=beta)
    return tf.nn.relu(x, name=name)


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size,
                                            align_corners=True)


def adaptive_avg_pooling(x):
    # global average pooling
    gap = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    return gap


class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 3), 'inputA'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 3), 'inputB')]

    @staticmethod
    def build_res_block(x, name, chan, first=False):
        with tf.variable_scope(name), \
                argscope([tf.layers.conv2d], kernel_size=3, strides=1,
                         padding='VALID'):
            input = x
            x = tpad(x, pad=1, mode='SYMMETRIC')
            x = tf.layers.conv2d(x, chan, activation=INReLU, name='conv0')
            x = tpad(x, pad=1, mode='SYMMETRIC')
            x = tf.layers.conv2d(x, chan, activation=tf.identity, name='conv1')
            x = InstanceNorm('inorm', x)
            return x + input

    @staticmethod
    def build_AdaIN_res_block(x, name, chan, mu, sigma, first=False):
        with tf.variable_scope(name), \
                argscope([tf.layers.conv2d], kernel_size=3, strides=1,
                         padding='VALID'):
            input = x
            activ = partial(AdaIN, gamma=mu, beta=sigma)
            x = tpad(x, pad=1, mode='SYMMETRIC')
            x = tf.layers.conv2d(x, chan, activation=activ, name='conv0')
            x = tf.nn.relu(x)
            x = tpad(x, pad=1, mode='SYMMETRIC')
            x = tf.layers.conv2d(x, chan, activation=activ, name='conv1')
            return x + input

    @auto_reuse_variable_scope
    def generator(self, content, style):
        l = content
        channel = pow(2, N_SAMPLE) * NF
        with tf.variable_scope('igen'):
            mu, sigma = self.MLP(style)

            for k in range(N_RES):
                l = Model.build_AdaIN_res_block(l, 'res{}'.format(k), channel,
                                                mu, sigma, first=(k == 0))
            x = l
            for i in range(N_SAMPLE):
                # IN removes the original feature mean and variance that represent important style information
                x = up_sample(x, scale_factor=2)
                x = tpad(x, 2, mode='reflect')
                x = tf.layers.conv2d(x, channel // 2, kernel_size=5, strides=1, name='conv_%d' % i)
                x = LayerNorm('lnorm_%d' % i, x)
                x = tf.nn.relu(x)
                channel = channel // 2
            l = tpad(x, 3, mode='SYMMETRIC')
            l = tf.layers.conv2d(l, 3, kernel_size=7, strides=1,
                                 activation=tf.tanh, name='G_logit')
        return l

    @auto_reuse_variable_scope
    def style_encoder(self, x):
        chan = NF
        with tf.variable_scope('senc'), \
            argscope([tf.layers.conv2d, Conv2D],
                     activation=tf.nn.relu, kernel_size=4, strides=2):

            x = tpad(x, pad=3, mode='reflect')
            x = tf.layers.conv2d(x, chan, kernel_size=7, strides=1,
                                 name='conv_0')

            for i in range(2):
                x = tpad(x, pad=1, mode='reflect')
                x = tf.layers.conv2d(x, chan * 2, name='conv_%d' % (i + 1))
                chan *= 2

            for i in range(2):
                x = tpad(x, pad=1, mode='reflect')
                x = tf.layers.conv2d(x, chan, name='dconv_%d' % i)

            x = adaptive_avg_pooling(x)  # global average pooling
            x = tf.layers.conv2d(x, STYLE_DIM, kernel_size=1, strides=1,
                                 activation=None, name='SE_logit')
        return x

    @auto_reuse_variable_scope
    def content_encoder(self, x):
        chan = NF
        with tf.variable_scope('cenc'), argscope([tf.layers.conv2d], activation=INReLU):
            x = tpad(x, pad=3, mode='reflect')
            x = tf.layers.conv2d(x, chan, kernel_size=7, strides=1, name='conv_0')

            for i in range(N_SAMPLE):
                x = tpad(x, 1, mode='reflect')
                x = tf.layers.conv2d(x, chan * 2, kernel_size=4, strides=2,
                                     name='conv_%d' % (i + 1))
                chan *= 2

            for i in range(N_RES):
                x = Model.build_res_block(x, 'res%d' % i, chan, first=(i == 0))
        return x

    @auto_reuse_variable_scope
    def MLP(self, style, name='MLP'):
        channel = pow(2, N_SAMPLE) * NF
        with tf.variable_scope(name), \
                argscope([tf.layers.dense],
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
            x = tf.layers.dense(tf.layers.flatten(style, name='sflat'), channel,
                                activation=tf.nn.relu, name='linear_0')
            x = tf.layers.dense(x, channel,
                                activation=tf.nn.relu, name='linear_1')

            mu = tf.layers.dense(x, channel, name='mu_')
            sigma = tf.layers.dense(x, channel, name='sigma_')

            mu = tf.reshape(mu, shape=[-1, 1, 1, channel], name='mu')
            sigma = tf.reshape(sigma, shape=[-1, 1, 1, channel], name='sigma')
        return mu, sigma

    @auto_reuse_variable_scope
    def discriminator(self, img):
        D_logit = []
        x_init = img
        with argscope([tf.layers.conv2d],
                      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                      activation=tf.nn.leaky_relu, kernel_size=4, strides=2):
            for scale in range(N_SCALE):
                chan = NF

                x = tpad(x_init, 1, mode='reflect')
                x = tf.layers.conv2d(x, chan,
                                     activation=tf.nn.leaky_relu,
                                     name='ms_%d_conv0' % scale)

                for i in range(1, N_DIS):
                    x = tpad(x, 1, mode='reflect')
                    x = tf.layers.conv2d(x, chan * 2,
                                         activation=tf.nn.leaky_relu,
                                         name='ms_%d_conv_%d' % (scale, i))
                    chan *= 2

                x = tf.layers.conv2d(x, 1, kernel_size=1, strides=1,
                                     name='ms_%d_D_logit' % scale)
                D_logit.append(x)

                x_init = tpad(x_init, 1, mode='reflect')
                x_init = tf.layers.average_pooling2d(x_init, pool_size=3,
                                                     strides=2, name='downsample_%d' % scale)
        return D_logit

    def build_graph(self, A, B):
        with tf.name_scope('preprocess'):
            A = A / 128.0 - 1.0
            B = B / 128.0 - 1.0
        with tf.name_scope('styleIn'):

            style_shape = [tf.shape(A)[0], 1, 1, STYLE_DIM]
            p_shape = [None, 1, 1, STYLE_DIM]
            styleA = tf.placeholder_with_default(
                tf.random_normal(style_shape,
                                 mean=0.0, stddev=1.0,
                                 dtype=tf.float32, name='styleArand'),
                shape=p_shape, name='styleA')
            styleB = tf.placeholder_with_default(
                tf.random_normal(style_shape,
                                 mean=0.0, stddev=1.0,
                                 dtype=tf.float32, name='styleBrand'),
                shape=p_shape, name='styleB')

        def vizN(name, a):
            with tf.name_scope(name):
                im = tf.concat(a, axis=2)
                im = (im + 1.0) * 128
                im = tf.clip_by_value(im, 0, 255)
                im = tf.cast(im, tf.uint8, name='viz')
            tf.summary.image(name, im, max_outputs=50)

        # use the initializers from torch
        with argscope([tf.layers.conv2d, tf.layers.dense],
                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer()), \
                argscope([InstanceNorm], use_affine=False):
            with tf.variable_scope('gen'):
                with tf.variable_scope('A'):
                    As = self.style_encoder(A)
                    Ac = self.content_encoder(A)
                with tf.variable_scope('B'):
                    Bs = self.style_encoder(B)
                    Bc = self.content_encoder(B)
                    # Within domain
                    BB = self.generator(Bc, Bs)
                    # Cross domain
                    AB = self.generator(Ac, styleB)
                    AB_swap = self.generator(Ac, styleA)

                    # re-encode
                    Ac_ = self.content_encoder(AB)
                    styleB_ = self.style_encoder(AB)
                with tf.variable_scope('A'):
                    # Within domain
                    AA = self.generator(Ac, As)
                    # Cross domain
                    BA = self.generator(Bc, styleA)
                    BA_swap = self.generator(Bc, styleB)

                    # re-encode
                    Bc_ = self.content_encoder(BA)
                    styleA_ = self.style_encoder(BA)
                if USE_CYCLE:
                    # Cycle back
                    with tf.variable_scope('A'):
                        ABA = self.generator(Ac_, As)

                    with tf.variable_scope('B'):
                        BAB = self.generator(Bc_, Bs)

            vizN('A_recon', [A, AB_swap, AB, AA])
            vizN('B_recon', [B, BA_swap, BA, BB])

            with tf.variable_scope('discrim'):
                with tf.variable_scope('A'):
                    A_dis_real = self.discriminator(A)
                    A_dis_fake = self.discriminator(BA)

                with tf.variable_scope('B'):
                    B_dis_real = self.discriminator(B)
                    B_dis_fake = self.discriminator(AB)

        def LSGAN_losses(real, fake):
            d_real = tf.reduce_mean(tf.squared_difference(real, 1), name='d_real')
            d_fake = tf.reduce_mean(tf.square(fake), name='d_fake')
            d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')

            g_loss = tf.reduce_mean(tf.squared_difference(fake, 1), name='g_loss')
            add_moving_summary(g_loss, d_loss)
            return g_loss, d_loss

        with tf.name_scope('losses'):
            with tf.name_scope('LossA'):
                # reconstruction loss
                if USE_CYCLE:
                    recon_loss_A = tf.reduce_mean(tf.abs(A - ABA),
                                                  name='recon_loss')
                else:
                    recon_loss_A = tf.constant(0.0, name='recon_loss')
                recon_loss_AA = tf.reduce_mean(tf.abs(A - AA),
                                               name='identity_loss')
                recon_loss_Astyle = tf.reduce_mean(tf.abs(styleA - styleA_),
                                                   name='style_loss')
                recon_loss_Acontent = tf.reduce_mean(tf.abs(Ac - Ac_),
                                                     name='content_loss')
                # gan loss
                G_loss_A, D_loss_A = zip(*[LSGAN_losses(real, fake) for
                                           real, fake in zip(A_dis_real, A_dis_fake)])
                G_loss_A = tf.add_n(G_loss_A, name='G_loss_msc')
                D_loss_A = tf.add_n(D_loss_A, name='D_loss_msc')

            with tf.name_scope('LossB'):
                if USE_CYCLE:
                    recon_loss_B = tf.reduce_mean(tf.abs(B - BAB),
                                                  name='recon_loss')
                else:
                    recon_loss_B = tf.constant(0.0, name='recon_loss')

                recon_loss_BB = tf.reduce_mean(tf.abs(B - BB),
                                               name='identity_loss')
                recon_loss_Bstyle = tf.reduce_mean(tf.abs(styleB - styleB_),
                                                   name='style_loss')
                recon_loss_Bcontent = tf.reduce_mean(tf.abs(Bc - Bc_),
                                                     name='content_loss')
                G_loss_B, D_loss_B = zip(*[LSGAN_losses(real, fake) for
                                           real, fake in zip(B_dis_real, B_dis_fake)])
                G_loss_B = tf.add_n(G_loss_B, name='G_loss_msc')
                D_loss_B = tf.add_n(D_loss_B, name='D_loss_msc')

            LAMBDA = 10.0
            self.g_loss = tf.add_n([(G_loss_A + G_loss_B),
                                    (recon_loss_A + recon_loss_B) * LAMBDA,
                                    (recon_loss_AA + recon_loss_BB) * LAMBDA,
                                    (recon_loss_Astyle + recon_loss_Bstyle),
                                    (recon_loss_Acontent + recon_loss_Bcontent)], name='G_loss_total')
            self.d_loss = tf.add(D_loss_A, D_loss_B, name='D_loss_total')
        self.collect_variables('gen', 'discrim')

        if REGULARIZE:
            wd_g = regularize_cost('gen/.*/kernel', l2_regularizer(1e-4),
                                   name='G_regularize')
            wd_d = regularize_cost('discrim/.*/kernel', l2_regularizer(1e-4),
                                   name='D_regularize')

            self.g_loss = tf.add(self.g_loss, wd_g, name='G_loss_totalr')
            self.d_loss = tf.add(self.d_loss, wd_d, name='D_loss_totalr')
        else:
            wd_g = tf.constant(0.0, name='G_regularize')
            wd_d = tf.constant(0.0, name='D_regularize')

        add_moving_summary(recon_loss_A, recon_loss_B,
                           recon_loss_AA, recon_loss_BB,
                           recon_loss_Astyle, recon_loss_Bstyle,
                           recon_loss_Acontent, recon_loss_Bcontent,
                           self.g_loss, self.d_loss,
                           wd_g, wd_d)

    def optimizer(self):
        lr = tf.train.exponential_decay(learning_rate=2e-4,
                                        global_step=get_global_step_var(),
                                        decay_steps=100 * 1000,
                                        decay_rate=0.5,
                                        staircase=True,
                                        name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in
        # terminal but this time without moving average
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def get_data(datadir, isTrain=True):
    if isTrain:
        augs = [
            imgaug.Resize(int(SHAPE * 1.12)),
            imgaug.RandomCrop(SHAPE),
            imgaug.Flip(horiz=True),
        ]
    else:
        augs = [imgaug.Resize(int(SHAPE * 1.12)),
                imgaug.CenterCrop(SHAPE)]

    def get_image_pairs(dir1, dir2):
        def get_df(dir):
            def glob_dir(ext):
                return glob.glob(os.path.join(dir, ext))
            files = sorted(glob_dir('*.jpg') + glob_dir('*.png'))
            df = ImageFromFile(files, channel=3, shuffle=isTrain)
            return AugmentImageComponent(df, augs)
        return JoinData([get_df(dir1), get_df(dir2)])

    names = ['trainA', 'trainB'] if isTrain else ['testA', 'testB']
    df = get_image_pairs(*[os.path.join(datadir, n) for n in names])
    df = BatchData(df, BATCH if isTrain else TEST_BATCH, remainder=not isTrain)
    df = PrefetchDataZMQ(df, 8 if isTrain else 1)
    return df


class VisualizeTestSet(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['inputA', 'inputB'], ['A_recon/viz', 'B_recon/viz'])

    def _before_train(self):
        global args
        self.val_ds = get_data(args.data, isTrain=False)
        self.val_ds.reset_state()

    def _trigger(self):
        idx = 0
        for iA, iB in self.val_ds:
            vizA, vizB = self.pred(iA, iB)
            self.trainer.monitors.put_image('testA-{}'.format(idx), vizA)
            self.trainer.monitors.put_image('testB-{}'.format(idx), vizB)
            idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', required=True,
        help='the image directory. should contain trainA/trainB/testA/testB')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    nr_tower = max(get_num_gpu(), 1)
    BATCH = BATCH // nr_tower
    logger.auto_set_dir(name=os.environ.get('JOB_ID'))

    df = get_data(args.data)
    df = PrintData(df)
    data = QueueInput(df)

    GANTrainer(data, Model(), num_gpu=nr_tower).train_with_defaults(
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=25),
            PeriodicTrigger(VisualizeTestSet(), every_k_epochs=3),
        ],
        max_epoch=1000,
        steps_per_epoch=min(data.size(), 1000),
        session_init=SaverRestore(args.load) if args.load else None
    )
