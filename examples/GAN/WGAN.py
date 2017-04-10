#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: WGAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import argparse

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.globvars import globalns as G
import tensorflow as tf
from GAN import SeparateGANTrainer

"""
Wasserstein-GAN.
See the docstring in DCGAN.py for usage.
"""

# Don't want to mix two examples together, but want to reuse the code.
# So here just import stuff from DCGAN, and change the batch size & model
import DCGAN
G.BATCH = 64


# a hacky way to change loss & optimizer of another script
class Model(DCGAN.Model):
    # def generator(self, z):
    # you can override generator to remove BatchNorm, it will still work in WGAN

    def build_losses(self, vecpos, vecneg):
        # the Wasserstein-GAN losses
        self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
        add_moving_summary(self.d_loss, self.g_loss)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)
        opt = tf.train.RMSPropOptimizer(lr)

        # add clipping to D optimizer
        def clip(p):
            n = p.op.name
            if not n.startswith('discrim/'):
                return None
            logger.info("Clip {}".format(n))
            return tf.clip_by_value(p, -0.01, 0.01)
        return optimizer.VariableAssignmentOptimizer(opt, clip)


DCGAN.Model = Model


if __name__ == '__main__':
    args = DCGAN.get_args()

    if args.sample:
        DCGAN.sample(args.load)
    else:
        assert args.data
        logger.auto_set_dir()
        config = DCGAN.get_config()
        config.steps_per_epoch = 500

        if args.load:
            config.session_init = SaverRestore(args.load)
        """
        The original code uses a different schedule.
        """
        SeparateGANTrainer(config, d_period=3).train()
