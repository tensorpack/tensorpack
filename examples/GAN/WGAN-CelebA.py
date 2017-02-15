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
        return tf.train.RMSPropOptimizer(lr)


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
    def __init__(self, config):
        self._input_method = QueueInput(config.dataflow)
        super(WGANTrainer, self).__init__(config)

    def _setup(self):
        super(WGANTrainer, self)._setup()
        self.build_train_tower()

        # add clipping to D optimizer
        def clip(p):
            n = p.op.name
            logger.info("Clip {}".format(n))
            return tf.clip_by_value(p, -0.01, 0.01)
        opt_G = self.model.get_optimizer()
        opt_D = optimizer.VariableAssignmentOptimizer(opt_G, clip)

        self.d_min = opt_D.minimize(
            self.model.d_loss, var_list=self.model.d_vars, name='d_min')
        self.g_min = opt_G.minimize(
            self.model.g_loss, var_list=self.model.g_vars, name='g_op')

    def run_step(self):
        for k in range(5):
            self.sess.run(self.d_min)
        ret = self.sess.run([self.g_min] + self.get_extra_fetches())
        return ret[1:]


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
