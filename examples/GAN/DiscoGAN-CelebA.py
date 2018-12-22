#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DiscoGAN-CelebA.py
# Author: Yuxin Wu

import argparse
import numpy as np
import os
import tensorflow as tf
from six.moves import map, zip

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary

from GAN import GANModelDesc, SeparateGANTrainer

"""
1. Download "aligned&cropped" version of celebA to /path/to/img_align_celeba.
2. Put list_attr_celeba.txt into that directory as well.
3. Start training gender transfer:
    ./DiscoGAN-CelebA.py --data /path/to/img_align_celeba --style-A Male
4. Visualize the gender conversion images in tensorboard.
"""

SHAPE = 64
BATCH = 64
NF = 64  # channel size


def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)


class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 3), 'inputA'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 3), 'inputB')]

    @auto_reuse_variable_scope
    def generator(self, img):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose],
                      activation=BNLReLU, kernel_size=4, strides=2), \
                argscope(Conv2DTranspose, activation=BNReLU):
            l = (LinearWrap(img)
                 .Conv2D('conv0', NF, activation=tf.nn.leaky_relu)
                 .Conv2D('conv1', NF * 2)
                 .Conv2D('conv2', NF * 4)
                 .Conv2D('conv3', NF * 8)
                 .Conv2DTranspose('deconv0', NF * 4)
                 .Conv2DTranspose('deconv1', NF * 2)
                 .Conv2DTranspose('deconv2', NF * 1)
                 .Conv2DTranspose('deconv3', 3, activation=tf.identity)
                 .tf.sigmoid()())
        return l

    @auto_reuse_variable_scope
    def discriminator(self, img):
        with argscope(Conv2D, activation=BNLReLU, kernel_size=4, strides=2):
            l = Conv2D('conv0', img, NF, activation=tf.nn.leaky_relu)
            relu1 = Conv2D('conv1', l, NF * 2)
            relu2 = Conv2D('conv2', relu1, NF * 4)
            relu3 = Conv2D('conv3', relu2, NF * 8)
            logits = FullyConnected('fc', relu3, 1, activation=tf.identity)
        return logits, [relu1, relu2, relu3]

    def get_feature_match_loss(self, feats_real, feats_fake):
        losses = []
        for real, fake in zip(feats_real, feats_fake):
            loss = tf.reduce_mean(tf.squared_difference(
                tf.reduce_mean(real, 0),
                tf.reduce_mean(fake, 0)),
                name='mse_feat_' + real.op.name)
            losses.append(loss)
        ret = tf.add_n(losses, name='feature_match_loss')
        add_moving_summary(ret)
        return ret

    def build_graph(self, A, B):
        A = tf.transpose(A / 255.0, [0, 3, 1, 2])
        B = tf.transpose(B / 255.0, [0, 3, 1, 2])

        # use the torch initializers
        with argscope([Conv2D, Conv2DTranspose, FullyConnected],
                      kernel_initializer=tf.variance_scaling_initializer(scale=0.333, distribution='uniform'),
                      use_bias=False), \
                argscope(BatchNorm, gamma_init=tf.random_uniform_initializer()), \
                argscope([Conv2D, Conv2DTranspose, BatchNorm], data_format='NCHW'):
            with tf.variable_scope('gen'):
                with tf.variable_scope('B'):
                    AB = self.generator(A)
                with tf.variable_scope('A'):
                    BA = self.generator(B)
                    ABA = self.generator(AB)
                with tf.variable_scope('B'):
                    BAB = self.generator(BA)

            viz_A_recon = tf.concat([A, AB, ABA], axis=3, name='viz_A_recon')
            viz_B_recon = tf.concat([B, BA, BAB], axis=3, name='viz_B_recon')
            tf.summary.image('Arecon', tf.transpose(viz_A_recon, [0, 2, 3, 1]), max_outputs=50)
            tf.summary.image('Brecon', tf.transpose(viz_B_recon, [0, 2, 3, 1]), max_outputs=50)

            with tf.variable_scope('discrim'):
                with tf.variable_scope('A'):
                    A_dis_real, A_feats_real = self.discriminator(A)
                    A_dis_fake, A_feats_fake = self.discriminator(BA)

                with tf.variable_scope('B'):
                    B_dis_real, B_feats_real = self.discriminator(B)
                    B_dis_fake, B_feats_fake = self.discriminator(AB)

        with tf.name_scope('LossA'):
            # reconstruction loss
            recon_loss_A = tf.reduce_mean(tf.squared_difference(A, ABA), name='recon_loss')
            # gan loss
            self.build_losses(A_dis_real, A_dis_fake)
            G_loss_A = self.g_loss
            D_loss_A = self.d_loss
            # feature matching loss
            fm_loss_A = self.get_feature_match_loss(A_feats_real, A_feats_fake)

        with tf.name_scope('LossB'):
            recon_loss_B = tf.reduce_mean(tf.squared_difference(B, BAB), name='recon_loss')
            self.build_losses(B_dis_real, B_dis_fake)
            G_loss_B = self.g_loss
            D_loss_B = self.d_loss
            fm_loss_B = self.get_feature_match_loss(B_feats_real, B_feats_fake)

        global_step = get_global_step_var()
        rate = tf.train.piecewise_constant(global_step, [np.int64(10000)], [0.01, 0.5])
        rate = tf.identity(rate, name='rate')   # TF issue#8594
        g_loss = tf.add_n([
            ((G_loss_A + G_loss_B) * 0.1 +
             (fm_loss_A + fm_loss_B) * 0.9) * (1 - rate),
            (recon_loss_A + recon_loss_B) * rate], name='G_loss_total')
        d_loss = tf.add_n([D_loss_A, D_loss_B], name='D_loss_total')

        self.collect_variables('gen', 'discrim')
        # weight decay
        wd_g = regularize_cost('gen/.*/W', l2_regularizer(1e-5), name='G_regularize')
        wd_d = regularize_cost('discrim/.*/W', l2_regularizer(1e-5), name='D_regularize')

        self.g_loss = g_loss + wd_g
        self.d_loss = d_loss + wd_d

        add_moving_summary(recon_loss_A, recon_loss_B, rate, g_loss, d_loss, wd_g, wd_d)

    def optimizer(self):
        return tf.train.AdamOptimizer(2e-4, beta1=0.5, epsilon=1e-3)


def get_celebA_data(datadir, styleA, styleB=None):
    def read_attr(attrfname):
        with open(attrfname) as f:
            nr_record = int(f.readline())
            headers = f.readline().strip().split()
            data = []
            for line in f:
                line = line.strip().split()[1:]
                line = list(map(int, line))
                assert len(line) == len(headers)
                data.append(line)
            assert len(data) == nr_record
            return headers, np.asarray(data, dtype='int8')

    headers, attrs = read_attr(os.path.join(datadir, 'list_attr_celeba.txt'))
    idxA = headers.index(styleA)
    listA = np.nonzero(attrs[:, idxA] == 1)[0]
    if styleB is not None:
        idxB = headers.index(styleB)
        listB = np.nonzero(attrs[:, idxB] == 1)[0]
    else:
        listB = np.nonzero(attrs[:, idxA] == -1)[0]

    def get_filelist(idxlist):
        return [os.path.join(datadir, '{:06d}.jpg'.format(x + 1))
                for x in idxlist]

    dfA = ImageFromFile(get_filelist(listA), channel=3, shuffle=True)
    dfB = ImageFromFile(get_filelist(listB), channel=3, shuffle=True)
    df = JoinData([dfA, dfB])
    augs = [
        imgaug.CenterCrop(160),
        imgaug.Resize(64)]
    df = AugmentImageComponents(df, augs, (0, 1))
    df = BatchData(df, BATCH)
    df = PrefetchDataZMQ(df, 3)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', required=True,
        help='the img_align_celeba directory. should also contain list_attr_celeba.txt')
    parser.add_argument('--style-A', help='style of A', default='Male')
    parser.add_argument('--style-B', help='style of B, default to "not A"')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    assert tf.test.is_gpu_available()
    logger.auto_set_dir()

    data = get_celebA_data(args.data, args.style_A, args.style_B)

    # train 1 D after 2 G
    SeparateGANTrainer(
        QueueInput(data), Model(), d_period=3).train_with_defaults(
        callbacks=[ModelSaver()],
        steps_per_epoch=300,
        max_epoch=250,
        session_init=SaverRestore(args.load) if args.load else None
    )
