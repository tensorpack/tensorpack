#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: WGAN.py
# Author: Yuxin Wu

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary

import DCGAN
from GAN import SeparateGANTrainer

"""
Wasserstein-GAN.
See the docstring in DCGAN.py for usage.
"""

# Don't want to mix two examples together, but want to reuse the code.
# So here just import stuff from DCGAN


class Model(DCGAN.Model):
    # def generator(self, z):
    # you can override generator to remove BatchNorm, it will still work in WGAN

    def build_losses(self, vecpos, vecneg):
        # the Wasserstein-GAN losses
        self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
        add_moving_summary(self.d_loss, self.g_loss)

    def optimizer(self):
        opt = tf.train.RMSPropOptimizer(1e-4)
        return opt

        # An alternative way to implement the clipping:
        """
        from tensorpack.tfutils import optimizer
        def clip(v):
            n = v.op.name
            if not n.startswith('discrim/'):
                return None
            logger.info("Clip {}".format(n))
            return tf.clip_by_value(v, -0.01, 0.01)
        return optimizer.VariableAssignmentOptimizer(opt, clip)
        """


class ClipCallback(Callback):
    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            n = v.op.name
            if not n.startswith('discrim/'):
                continue
            logger.info("Clip {}".format(n))
            ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
        self._op = tf.group(*ops, name='clip')

    def _trigger_step(self):
        self._op.run()


if __name__ == '__main__':
    args = DCGAN.get_args(default_batch=64)

    M = Model(shape=args.final_size, batch=args.batch, z_dim=args.z_dim)
    if args.sample:
        DCGAN.sample(M, args.load)
    else:
        logger.auto_set_dir()

        # The original code uses a different schedule, but this seems to work well.
        # Train 1 D after 2 G
        SeparateGANTrainer(
            input=QueueInput(DCGAN.get_data()),
            model=M, d_period=3).train_with_defaults(
            callbacks=[ModelSaver(), ClipCallback()],
            steps_per_epoch=500,
            max_epoch=200,
            session_init=SaverRestore(args.load) if args.load else None
        )
