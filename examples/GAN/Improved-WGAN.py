#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Improved-WGAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.globvars import globalns as G
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorflow as tf

from GAN import SeparateGANTrainer

"""
Improved Wasserstein-GAN.
See the docstring in DCGAN.py for usage.
"""

# Don't want to mix two examples together, but want to reuse the code.
# So here just import stuff from DCGAN, and change the batch size & model
import DCGAN
G.BATCH = 64
G.Z_DIM = 128


class Model(DCGAN.Model):
    # replace BatchNorm by LayerNorm
    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        nf = 64
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
                argscope(LeakyReLU, alpha=0.2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf, nl=LeakyReLU)
                 .Conv2D('conv1', nf * 2)
                 .LayerNorm('ln1').LeakyReLU()
                 .Conv2D('conv2', nf * 4)
                 .LayerNorm('ln2').LeakyReLU()
                 .Conv2D('conv3', nf * 8)
                 .LayerNorm('ln3').LeakyReLU()
                 .FullyConnected('fct', 1, nl=tf.identity)())
        return tf.reshape(l, [-1])

    def _build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1

        z = tf.random_normal([G.BATCH, G.Z_DIM], name='z_train')
        z = tf.placeholder_with_default(z, [None, G.Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z)
            tf.summary.image('generated-samples', image_gen, max_outputs=30)

            alpha = tf.random_uniform(shape=[G.BATCH, 1, 1, 1],
                                      minval=0., maxval=1., name='alpha')
            interp = image_pos + alpha * (image_gen - image_pos)

            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos)
                vecneg = self.discriminator(image_gen)
                vec_interp = self.discriminator(interp)

        # the Wasserstein-GAN losses
        self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')

        # the gradient penalty loss
        gradients = tf.gradients(vec_interp, [interp])[0]
        gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
        gradients_rms = symbolic_functions.rms(gradients, 'gradient_rms')
        gradient_penalty = tf.reduce_mean(tf.square(gradients - 1), name='gradient_penalty')
        add_moving_summary(self.d_loss, self.g_loss, gradient_penalty, gradients_rms)

        self.d_loss = tf.add(self.d_loss, 10 * gradient_penalty)

        self.collect_variables()

    def _get_optimizer(self):
        opt = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9)
        return opt


if __name__ == '__main__':
    args = DCGAN.get_args()
    if args.sample:
        DCGAN.sample(Model(), args.load)
    else:
        assert args.data
        logger.auto_set_dir()
        SeparateGANTrainer(
            QueueInput(DCGAN.get_data(args.data)),
            Model(), g_period=6).train_with_defaults(
            callbacks=[ModelSaver()],
            steps_per_epoch=300,
            max_epoch=200,
            session_init=SaverRestore(args.load) if args.load else None
        )
