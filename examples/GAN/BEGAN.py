#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: BEGAN.py
# Author: Yuxin Wu

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.gpu import get_num_gpu

import DCGAN
from GAN import GANModelDesc, GANTrainer, MultiGPUGANTrainer

"""
Boundary Equilibrium GAN.
See the docstring in DCGAN.py for usage.

A pretrained model on CelebA is at http://models.tensorpack.com/GAN/
"""


NH = 64
NF = 64
GAMMA = 0.5


class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, args.final_size, args.final_size, 3), 'input')]

    @auto_reuse_variable_scope
    def decoder(self, z):
        l = FullyConnected('fc', z, NF * 8 * 8)
        l = tf.reshape(l, [-1, 8, 8, NF])

        with argscope(Conv2D, activation=tf.nn.elu, kernel_size=3, strides=1):
            l = (LinearWrap(l)
                 .Conv2D('conv1.1', NF)
                 .Conv2D('conv1.2', NF)
                 .tf.image.resize_nearest_neighbor([16, 16], align_corners=True)
                 .Conv2D('conv2.1', NF)
                 .Conv2D('conv2.2', NF)
                 .tf.image.resize_nearest_neighbor([32, 32], align_corners=True)
                 .Conv2D('conv3.1', NF)
                 .Conv2D('conv3.2', NF)
                 .tf.image.resize_nearest_neighbor([64, 64], align_corners=True)
                 .Conv2D('conv4.1', NF)
                 .Conv2D('conv4.2', NF)
                 .Conv2D('conv4.3', 3, activation=tf.identity)())
        return l

    @auto_reuse_variable_scope
    def encoder(self, imgs):
        with argscope(Conv2D, activation=tf.nn.elu, kernel_size=3, strides=1):
            l = (LinearWrap(imgs)
                 .Conv2D('conv1.1', NF)
                 .Conv2D('conv1.2', NF)
                 .Conv2D('conv1.3', NF * 2)
                 .AvgPooling('pool1', 2)
                 # 32
                 .Conv2D('conv2.1', NF * 2)
                 .Conv2D('conv2.2', NF * 3)
                 .AvgPooling('pool2', 2)
                 # 16
                 .Conv2D('conv3.1', NF * 3)
                 .Conv2D('conv3.2', NF * 4)
                 .AvgPooling('pool3', 2)
                 # 8
                 .Conv2D('conv4.1', NF * 4)
                 .Conv2D('conv4.2', NF * 4)

                 .FullyConnected('fc', NH)())
        return l

    def build_graph(self, image_pos):
        image_pos = image_pos / 128.0 - 1

        z = tf.random_uniform([args.batch, args.z_dim], minval=-1, maxval=1, name='z_train')
        z = tf.placeholder_with_default(z, [None, args.z_dim], name='z')

        def summary_image(name, x):
            x = (x + 1.0) * 128.0
            x = tf.clip_by_value(x, 0, 255)
            tf.summary.image(name, tf.cast(x, tf.uint8), max_outputs=30)

        with argscope([Conv2D, FullyConnected],
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.decoder(z)

            with tf.variable_scope('discrim'):
                with tf.variable_scope('enc'):
                    hidden_pos = self.encoder(image_pos)
                    hidden_neg = self.encoder(image_gen)

                with tf.variable_scope('dec'):
                    recon_pos = self.decoder(hidden_pos)
                    recon_neg = self.decoder(hidden_neg)

        with tf.name_scope('viz'):
            summary_image('generated-samples', image_gen)
            summary_image('reconstruct-real', recon_pos)
            summary_image('reconstruct-fake', recon_neg)

        with tf.name_scope('losses'):
            L_pos = tf.reduce_mean(tf.abs(recon_pos - image_pos), name='loss_pos')
            L_neg = tf.reduce_mean(tf.abs(recon_neg - image_gen), name='loss_neg')

            eq = tf.subtract(GAMMA * L_pos, L_neg, name='equilibrium')
            measure = tf.add(L_pos, tf.abs(eq), name='measure')

            kt = tf.get_variable('kt', dtype=tf.float32, initializer=0.0)

            update_kt = kt.assign_add(1e-3 * eq)
            with tf.control_dependencies([update_kt]):
                self.d_loss = tf.subtract(L_pos, kt * L_neg, name='loss_D')
                self.g_loss = L_neg

        add_moving_summary(L_pos, L_neg, eq, measure, self.d_loss)
        tf.summary.scalar('kt', kt)

        self.collect_variables()

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
        return opt


if __name__ == '__main__':
    args = DCGAN.get_args(default_batch=32, default_z_dim=64)
    if args.sample:
        DCGAN.sample(Model(), args.load, 'gen/conv4.3/output')
    else:
        logger.auto_set_dir()

        input = QueueInput(DCGAN.get_data())
        model = Model()
        nr_tower = max(get_num_gpu(), 1)
        if nr_tower == 1:
            trainer = GANTrainer(input, model)
        else:
            trainer = MultiGPUGANTrainer(nr_tower, input, model)

        trainer.train_with_defaults(
            callbacks=[
                ModelSaver(),
                StatMonitorParamSetter(
                    'learning_rate', 'measure', lambda x: x * 0.5, 0, 10)
            ],
            session_init=SaverRestore(args.load) if args.load else None,
            steps_per_epoch=500, max_epoch=400)
