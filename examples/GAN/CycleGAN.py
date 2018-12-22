#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: CycleGAN.py
# Author: Yuxin Wu

import argparse
import glob
import os
import tensorflow as tf
from six.moves import range

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary

from GAN import GANModelDesc, GANTrainer

"""
1. Download the dataset following the original project: https://github.com/junyanz/CycleGAN#train
2. ./CycleGAN.py --data /path/to/datasets/horse2zebra
Training and testing visualizations will be in tensorboard.

This implementation doesn't use fake sample buffer.
It's not hard to add but I didn't observe any difference with it.
"""

SHAPE = 256
BATCH = 1
TEST_BATCH = 32
NF = 64  # channel size


def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)


class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 3), 'inputA'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 3), 'inputB')]

    @staticmethod
    def build_res_block(x, name, chan, first=False):
        with tf.variable_scope(name):
            input = x
            return (LinearWrap(x)
                    .tf.pad([[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
                    .Conv2D('conv0', chan, 3, padding='VALID')
                    .tf.pad([[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
                    .Conv2D('conv1', chan, 3, padding='VALID', activation=tf.identity)
                    .InstanceNorm('inorm')()) + input

    @auto_reuse_variable_scope
    def generator(self, img):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INReLU):
            l = (LinearWrap(img)
                 .tf.pad([[0, 0], [0, 0], [3, 3], [3, 3]], mode='SYMMETRIC')
                 .Conv2D('conv0', NF, 7, padding='VALID')
                 .Conv2D('conv1', NF * 2, 3, strides=2)
                 .Conv2D('conv2', NF * 4, 3, strides=2)())
            for k in range(9):
                l = Model.build_res_block(l, 'res{}'.format(k), NF * 4, first=(k == 0))
            l = (LinearWrap(l)
                 .Conv2DTranspose('deconv0', NF * 2, 3, strides=2)
                 .Conv2DTranspose('deconv1', NF * 1, 3, strides=2)
                 .tf.pad([[0, 0], [0, 0], [3, 3], [3, 3]], mode='SYMMETRIC')
                 .Conv2D('convlast', 3, 7, padding='VALID', activation=tf.tanh, use_bias=True)())
        return l

    @auto_reuse_variable_scope
    def discriminator(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=4, strides=2):
            l = (LinearWrap(img)
                 .Conv2D('conv0', NF, activation=tf.nn.leaky_relu)
                 .Conv2D('conv1', NF * 2)
                 .Conv2D('conv2', NF * 4)
                 .Conv2D('conv3', NF * 8, strides=1)
                 .Conv2D('conv4', 1, strides=1, activation=tf.identity, use_bias=True)())
        return l

    def build_graph(self, A, B):
        with tf.name_scope('preprocess'):
            A = tf.transpose(A / 128.0 - 1.0, [0, 3, 1, 2])
            B = tf.transpose(B / 128.0 - 1.0, [0, 3, 1, 2])

        def viz3(name, a, b, c):
            with tf.name_scope(name):
                im = tf.concat([a, b, c], axis=3)
                im = tf.transpose(im, [0, 2, 3, 1])
                im = (im + 1.0) * 128
                im = tf.clip_by_value(im, 0, 255)
                im = tf.cast(im, tf.uint8, name='viz')
            tf.summary.image(name, im, max_outputs=50)

        # use the initializers from torch
        with argscope([Conv2D, Conv2DTranspose], use_bias=False,
                      kernel_initializer=tf.random_normal_initializer(stddev=0.02)), \
                argscope([Conv2D, Conv2DTranspose, InstanceNorm], data_format='channels_first'):
            with tf.variable_scope('gen'):
                with tf.variable_scope('B'):
                    AB = self.generator(A)
                with tf.variable_scope('A'):
                    BA = self.generator(B)
                    ABA = self.generator(AB)
                with tf.variable_scope('B'):
                    BAB = self.generator(BA)

            viz3('A_recon', A, AB, ABA)
            viz3('B_recon', B, BA, BAB)

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
                recon_loss_A = tf.reduce_mean(tf.abs(A - ABA), name='recon_loss')
                # gan loss
                G_loss_A, D_loss_A = LSGAN_losses(A_dis_real, A_dis_fake)

            with tf.name_scope('LossB'):
                recon_loss_B = tf.reduce_mean(tf.abs(B - BAB), name='recon_loss')
                G_loss_B, D_loss_B = LSGAN_losses(B_dis_real, B_dis_fake)

            LAMBDA = 10.0
            self.g_loss = tf.add((G_loss_A + G_loss_B),
                                 (recon_loss_A + recon_loss_B) * LAMBDA, name='G_loss_total')
            self.d_loss = tf.add(D_loss_A, D_loss_B, name='D_loss_total')
        self.collect_variables('gen', 'discrim')

        add_moving_summary(recon_loss_A, recon_loss_B, self.g_loss, self.d_loss)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def get_data(datadir, isTrain=True):
    if isTrain:
        augs = [
            imgaug.Resize(int(SHAPE * 1.12)),
            imgaug.RandomCrop(SHAPE),
            imgaug.Flip(horiz=True),
        ]
    else:
        augs = [imgaug.Resize(SHAPE)]

    def get_image_pairs(dir1, dir2):
        def get_df(dir):
            files = sorted(glob.glob(os.path.join(dir, '*.jpg')))
            df = ImageFromFile(files, channel=3, shuffle=isTrain)
            return AugmentImageComponent(df, augs)
        return JoinData([get_df(dir1), get_df(dir2)])

    names = ['trainA', 'trainB'] if isTrain else ['testA', 'testB']
    df = get_image_pairs(*[os.path.join(datadir, n) for n in names])
    df = BatchData(df, BATCH if isTrain else TEST_BATCH)
    df = PrefetchDataZMQ(df, 2 if isTrain else 1)
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

    logger.auto_set_dir()

    df = get_data(args.data)
    df = PrintData(df)
    data = QueueInput(df)

    GANTrainer(data, Model()).train_with_defaults(
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter(
                'learning_rate',
                [(100, 2e-4), (200, 0)], interp='linear'),
            PeriodicTrigger(VisualizeTestSet(), every_k_epochs=3),
        ],
        max_epoch=195,
        steps_per_epoch=data.size(),
        session_init=SaverRestore(args.load) if args.load else None
    )
