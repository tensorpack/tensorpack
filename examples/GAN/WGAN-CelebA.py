#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: WGAN-CelebA.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import os
import argparse

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from GAN import GANTrainer

"""
Wasserstein-GAN.
See the docstring in DCGAN-CelebA.py for usage.

Actually, just using the clip is enough for WGAN to work (even without BN in generator).
The wasserstein loss is not the key factor.
"""

# Don't want to mix two examples together, but want to reuse the code.
# So here just import stuff from DCGAN-CelebA, and change the batch size & model
import imp
DCGAN = imp.load_source(
    'DCGAN',
    os.path.join(os.path.dirname(__file__), 'DCGAN-CelebA.py'))


class Model(DCGAN.Model):
    # def generator(self, z):
    # you can override generator to remove BatchNorm, it will still work in WGAN

    def build_losses(self, vecpos, vecneg):
        # the Wasserstein-GAN losses
        self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        self.g_loss = -tf.reduce_mean(vecneg, name='g_loss')
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


DCGAN.BATCH = 64
DCGAN.Model = Model


def get_config():
    return TrainConfig(
        model=Model(),
        # use the same data in the DCGAN example
        dataflow=DCGAN.get_data(args.data),
        callbacks=[ModelSaver()],
        session_config=get_default_sess_config(0.5),
        steps_per_epoch=300,
        max_epoch=200,
    )


class WGANTrainer(FeedfreeTrainerBase):
    """ A new trainer which runs two optimization ops with 5:1 ratio.
        This is to be consistent with the original code, but I found just
        running them 1:1 (i.e. just using the existing GANTrainer) also works well.
    """
    def __init__(self, config):
        self._input_method = QueueInput(config.dataflow)
        super(WGANTrainer, self).__init__(config)

    def _setup(self):
        super(WGANTrainer, self)._setup()
        self.build_train_tower()

        opt = self.model.get_optimizer()
        self.d_min = opt.minimize(
            self.model.d_loss, var_list=self.model.d_vars, name='d_min')
        self.g_min = opt.minimize(
            self.model.g_loss, var_list=self.model.g_vars, name='g_op')

    def run_step(self):
        for k in range(5):
            self.hooked_sess.run(self.d_min)
        self.hooked_sess.run(self.g_min)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--data', help='a jpeg directory')
    args = parser.parse_args()
    if args.sample:
        DCGAN.sample(args.load)
    else:
        assert args.data
        logger.auto_set_dir()
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        WGANTrainer(config).train()
